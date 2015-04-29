###############################################################################
#   lazyflow: data flow based lazy parallel computation framework
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
#          http://ilastik.org/license/
###############################################################################

import numpy as np
import vigra
import h5py
from threading import Lock as HardLock

from collections import defaultdict
from functools import partial, wraps
import itertools

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.roi import determine_optimal_request_blockshape
from lazyflow.request import RequestLock
from lazyflow.operators import OpReorderAxes
from lazyflow.operators.opCache import ObservableCache
from lazyflow.operators.opCompressedCache import OpCompressedCache

from lazyflow.operators.opLazyRegionGrowing import OpLazyRegionGrowing
from lazyflow.operators.opSimpleConnectedComponents import\
    OpSimpleConnectedComponents
from lazyflow.operators.opLazyRegionGrowing import threadsafe

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

Lock = RequestLock
_LABEL_TYPE = np.uint32


# Locking decorator similar to threadsafe() that locks per chunk. The
# first arguent of the wrapped method must be the chunk index.
def _chunksynchronized(method):
    @wraps(method)
    def synchronizedmethod(self, chunkIndex, *args, **kwargs):
        with self._chunk_locks[chunkIndex]:
            return method(self, chunkIndex, *args, **kwargs)
    return synchronizedmethod


# OpLazyConnectedComponents
# =========================
#
# This operator provides a connected components (labeling) algorithm
# that evaluates lazyly, i.e. you don't need to process a full volume
# for getting the connected components in some ROI. The operator just
# computes spatial connected components, channels and time slices are
# treated independently.
#
# Guarantees
# ==========
#
# The resulting label image is equivalent[1] to the label image of
# the vigra function acting on the whole volume. The output is
# guaranteed to have exactly one label per connected component, and to
# have a contiguous set of labels *over the entire spatial volume*. 
# This means that your first request to some subregion *could* give
# you labels [120..135], but somewhere else in the volume there will be
# at least one pixel labeled with each of [1..119]. Furthermore, the
# output is computed as lazy as possible, meaning that only the chunks
# that intersect with an object in the requested ROI are computed. The
# operator.execute() method is thread safe, but does not spawn new
# requests besides the ones needed for gathering input data.
# This operator conforms to OpLabelingABC, meaning that it can be used
# as an implementation backend in lazyflow.OpLabelVolume.
#
# [1] The labels might not be equal, but the uniquely labeled objects
#     are the same for both versions.
#
# Parallelization
# ===============
#
# The operator is thread safe, but has no parallelization of its own.
# The user (the GUI) is responsible for tiling the volume and spawning
# parallel requests to the operator's output slots.
#
# Implementation Details
# ======================
#
# The connected components algorithm used internally (chunk wise) is
#       vigra.labelMultiArrayWithBackground()
# with the default 4/6-neighborhood. See 
#       http://ukoethe.github.io/vigra/doc/vigra/group__Labeling.html
# for details.
#
# There are 3 kinds of labels that we need to consider throughout the operator:
#     * local labels: The output of the chunk wise labeling calls. These are
#       stored in self._cache, a compressed VigraArray.
#       aka 'local'
#     * global indices: The mapping of local labels to unique global indices.
#       The actual implemetation is hidden in self.localToGlobal().
#       aka 'global'
#     * global labels: The final labels that are communicated to the outside
#       world. These must be contiguous, i.e. if  global label j appears in the
#       output, then, for every global label i<j, i also appears in the output.
#       The actual implementation is hidden in self.globalToFinal().
#       aka 'final'
#
# The strategy we are using could be written as the following piece of
# pseudocode:
#   - put all requested chunks in the processing queue
#   - for each chunk in processing queue:
#       * label the chunk, store the labels
#       * for each neighbour of chunk:
#           + identify objects extending to this neighbour, call makeUnion()
#           + if there were such objects, put chunk in processing queue
#
# In addition to this short algorithm, there is some bookkeeping going
# on that allows us to map the different label types to one another,
# and avoids excessive computation by tracking which process is 
# responsible for which particular set of local labels.
#
class OpLazyConnectedComponents(OpLazyRegionGrowing, ObservableCache):
    name = "OpLazyConnectedComponents"
    supportedDtypes = [np.uint8, np.uint32, np.float32]

    # input data (usually segmented)
    Input = InputSlot()

    # background with axes 'txyzc', spatial axes must be singletons
    # (this layout is needed to be compatible with OpLabelVolume)
    Background = InputSlot(value=0)

    # the labeled output
    Output = OutputSlot()

    ### INTERNALS -- DO NOT USE ###
    _Labels = OutputSlot()
    _Input = OutputSlot()
    _Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpLazyConnectedComponents, self).__init__(*args, **kwargs)
        self._lock = Lock()

        # reordering operators - we want to handle txyzc inside this operator
        self._opIn = OpReorderAxes(parent=self)
        self._opIn.AxisOrder.setValue('txyzc')
        self._opIn.Input.connect(self.Input)
        self._Input.connect(self._opIn.Output)

        self._labeler = OpSimpleConnectedComponents(parent=self)
        self._labeler.Input.connect(self._Input)
        self._labeler.Background.connect(self.Background)
        self._labelCache = None

        self._opOut = OpReorderAxes(parent=self)
        self._opOut.Input.connect(self._Output)
        self.Output.connect(self._opOut.Output)
        
        # Now that we're initialized, it's safe to register with the memory manager
        self.registerWithMemoryManager()

    def setupOutputs(self):
        self._Output.meta.assignFrom(self._Input.meta)
        self._Output.meta.dtype = _LABEL_TYPE
        if not self.Input.meta.dtype in self.supportedDtypes:
            raise ValueError(
                "Cannot label data type {}".format(self.Input.meta.dtype))

        self._setDefaultInternals()
        super(OpLazyConnectedComponents, self).setupOutputs()

        # go back to original order
        self._opOut.AxisOrder.setValue(self.Input.meta.getAxisKeys())

        # set up label cache
        if self._labelCache is not None:
            self._Labels.disconnect()
            self._labelCache.Input.disconnect()
            self._labelCache = None

        self._labelCache = OpCompressedCache(parent=self)
        self._labelCache.Input.connect(self._labeler.Output)
        self._Labels.connect(self._labelCache.Output)
        self._labelCache.BlockShape.setValue(self._chunkShape)

    def execute(self, slot, subindex, roi, result):
        if slot is self._Output:
            logger.debug("Execute for {}".format(roi))
            self.executeRegionGrowing(roi, result)
        else:
            raise ValueError("Request to invalid slot {}".format(str(slot)))

    def propagateDirty(self, slot, subindex, roi):
        """
        Dirty handling is not trivial with this operator. The worst
        case happens when an object disappears entirely, meaning that
        the assigned labels would not be contiguous anymore. We could
        check for that here, and set everything dirty if it's the
        case, but this would require us to run the entire algorithm
        once again, which is not desireable in propagateDirty(). The
        simplest valid decision is to set the whole output dirty in
        every case.
        """
        # TODO make t, c slices independent.
        self._setDefaultInternals()
        #super(OpLazyConnectedComponents, self).setupOutputs()
        self.Output.setDirty(slice(None))

    def shape(self):
        return getattr(self, "_shape", None)

    def chunkShape(self):
        return getattr(self, "_chunkShape", None)

    @_chunksynchronized
    def handleSingleChunk(self, chunkIndex):
        if self._numIndices[chunkIndex] >= 0:
            # this chunk is already labeled
            return set()

        roi = self.chunkIndexToRoi(chunkIndex)
        logger.debug("requesting labels for chunk {} ({})".format(
            chunkIndex, roi))
        labeled = self._Labels.get(roi).wait()
        labeled = vigra.taggedView(labeled,
                                   axistags=self._Labels.meta.axistags)

        # update the labeling information        
        numLabels = labeled.max()  # we ignore 0 here
        self._numIndices[chunkIndex] = numLabels
        if numLabels > 0:
            with self._lock: 
                # determine the offset
                # localLabel + offset = globalLabel (for localLabel>0)
                offset = self._uf.makeNewIndex()
                self._globalLabelOffset[chunkIndex] = offset - 1

                # get n-1 more labels
                for i in range(numLabels-1):
                    self._uf.makeNewIndex()
        return set(range(1, numLabels+1))

    def mergeChunks(self, chunkA, chunkB):
        a, b = self.orderPair(chunkA, chunkB)
        reordered = a != chunkA
        workA, workB = self._merge(a, b)
        if reordered:
            return workB, workA
        else:
            return workA, workB

    @_chunksynchronized
    def _merge(self, chunkA, chunkB):
        """
        merge the labels of two adjacent chunks

        the chunks have to be ordered lexicographically, e.g. by self._orderPair
        """
        if chunkB in self._mergeMap[chunkA]:
            return set(), set()
        self._mergeMap[chunkA].append(chunkB)

        hyperplane_roi_a, hyperplane_roi_b = \
            self.chunkIndexToHyperplane(chunkA, chunkB)
        hyperplane_index_a = hyperplane_roi_a.toSlice()
        hyperplane_index_b = hyperplane_roi_b.toSlice()

        label_hyperplane_a = self._Labels.get(hyperplane_roi_a).wait()
        label_hyperplane_b = self._Labels.get(hyperplane_roi_b).wait()

        # see if we have border labels at all
        adjacent_bool_inds = np.logical_and(label_hyperplane_a > 0,
                                            label_hyperplane_b > 0)
        if not np.any(adjacent_bool_inds):
            return set(), set()

        # check if the labels do actually belong to the same component
        hyperplane_a = self._Input.get(hyperplane_roi_a).wait()
        hyperplane_b = self._Input.get(hyperplane_roi_b).wait()
        adjacent_bool_inds = np.logical_and(
            adjacent_bool_inds, hyperplane_a == hyperplane_b)

        # union find manipulations are critical
        with self._lock:
            map_a = self.localToGlobal(chunkA)
            map_b = self.localToGlobal(chunkB)
            labels_a = map_a[label_hyperplane_a[adjacent_bool_inds]]
            labels_b = map_b[label_hyperplane_b[adjacent_bool_inds]]
            for a, b in zip(labels_a, labels_b):
                assert a not in self._globalToFinal, "Invalid merge"
                assert b not in self._globalToFinal, "Invalid merge"
                self._uf.makeUnion(a, b)

            logger.debug("merged chunks {} and {}".format(chunkA, chunkB))
        correspondingLabelsA = label_hyperplane_a[adjacent_bool_inds]
        correspondingLabelsB = label_hyperplane_b[adjacent_bool_inds]
        return set(correspondingLabelsA), set(correspondingLabelsB)

    def fillResult(self, roi, result):
        assert np.all(roi.stop - roi.start == result.shape)

        logger.debug("mapping roi {}".format(roi))
        indices = self.roiToChunkIndex(roi)
        for idx in indices:
            newroi = self.chunkIndexToRoi(idx)
            newroi.stop = np.minimum(newroi.stop, roi.stop)
            newroi.start = np.maximum(newroi.start, roi.start)
            chunk = self._Labels.get(newroi).wait()
            final_chunk = self._mapChunk(idx, chunk)
            newroi.start -= roi.start
            newroi.stop -= roi.start
            result_slice = newroi.toSlice()
            result[result_slice] = final_chunk

    def _mapChunk(self, chunkIndex, local_labeling):
        """
        map local labels for this chunk to final labels
        """
        labels = self.localToGlobal(chunkIndex)
        labels = self.globalToFinal(chunkIndex[0], chunkIndex[4], labels)
        finalized_chunk = labels[local_labeling]
        return finalized_chunk

    def localToGlobal(self, chunkIndex):
        """
        get an array of global labels in use by this chunk

        This array can be used as a mapping via
            mapping = localToGlobal(...)
            mapped = mapping[locallyLabeledArray]
        The global labels are updated to their current state according
        to the global UnionFind structure.
        """
        offset = self._globalLabelOffset[chunkIndex]
        numLabels = self._numIndices[chunkIndex]
        labels = np.arange(1, numLabels+1, dtype=_LABEL_TYPE) + offset

        labels = np.asarray(map(self._uf.findIndex, labels),
                            dtype=_LABEL_TYPE)

        # we got 'numLabels' real labels, and one label '0', so our
        # output has to have numLabels+1 elements
        out = np.zeros((numLabels+1,), dtype=_LABEL_TYPE)
        out[1:] = labels
        return out

    def globalToFinal(self, t, c, labels):
        """
        map an array of global indices to final labels

        after calling this function, the labels passed in may not be
        used with UnionFind.makeUnion any more!
        """
        newlabels = labels.copy()
        d = self._globalToFinal[(t, c)]
        labeler = self._labelIterators[(t, c)]
        for k in np.unique(labels):
            l = self._uf.findIndex(k)
            if l == 0:
                continue

            if l not in d:
                nextLabel = labeler.next()
                d[l] = nextLabel
            newlabels[labels == k] = d[l]
        return newlabels

    ##########################################################################
    ##################### HELPER METHODS #####################################
    ##########################################################################

    def _setDefaultInternals(self):
        """
        fills attributes with standard values

        Must be called on each setupOutputs()
        """
        # chunk array shape calculation
        shape = self._Input.meta.shape
        assert len(shape) == 5
        chunkShape = self._automaticChunkShape()
        assert len(shape) == len(chunkShape),\
            "Encountered an invalid chunkShape"
        self._chunkShape = np.asarray(chunkShape, dtype=np.int)
        self._shape = shape

        ### global indices ###
        # offset (global labels - local labels) per chunk
        self._globalLabelOffset = defaultdict(partial(_LABEL_TYPE, 1))

        # keep track of number of indices in chunk (-1 == not labeled yet)
        self._numIndices = defaultdict(partial(np.int32, -1))

        # union find data structure, tells us for every global index to which
        # label it belongs
        self._uf = UnionFindArray(_LABEL_TYPE(1))

        ### global labels ###
        # keep track of assigned global labels
        gen = partial(InfiniteLabelIterator, 1, dtype=_LABEL_TYPE)
        self._labelIterators = defaultdict(gen)
        self._globalToFinal = defaultdict(dict)

        ### algorithmic ###

        # keep track of merged regions
        self._mergeMap = defaultdict(list)

        # locks that keep threads from changing a specific chunk
        self._chunk_locks = defaultdict(Lock)

    def _automaticChunkShape(self):
        """
        get chunk shape appropriate for input data
        """
        slot = self._Input
        if not slot.ready():
            return None

        # use about 10MiB per chunk
        ram = 10*1024**2
        ram_per_pixel = slot.meta.getDtypeBytes()

        def prepareShape(s):
            return (1,) + tuple(s)[1:4] + (1,)

        max_shape = prepareShape(slot.meta.shape)
        if slot.meta.ideal_blockshape is not None:
            ideal_shape = prepareShape(slot.meta.ideal_blockshape)
        else:
            ideal_shape = (1, 0, 0, 0, 1)

        if np.prod(ideal_shape) > 0:
            return np.minimum(ideal_shape, max_shape)

        chunkShape = determine_optimal_request_blockshape(
            max_shape, ideal_shape, ram_per_pixel, 1, ram)
        logger.debug("Using chunk shape {}".format(chunkShape))
        return chunkShape

    # ======== CACHE API ========

    def usedMemory(self):
        #TODO check administrative data
        return 0

    def fractionOfUsedMemoryDirty(self):
        # we do not handle dirtyness
        return 0.0

    def generateReport(self, report):
        super(OpLazyConnectedComponents, self).generateReport(report)
        report.dtype = _LABEL_TYPE


###########
#  TOOLS  #
###########


class UnionFindArray(object):
    """
    python implementation of vigra's UnionFindArray structure
    """

    def __init__(self, nextFree=1):
        self._map = dict(zip(*(xrange(nextFree),)*2))
        self._lock = Lock()
        self._nextFree = nextFree
        self._it = None

    @threadsafe
    def makeUnion(self, a, b):
        """
        join regions a and b
        """
        assert a in self._map
        assert b in self._map

        a = self._findIndex(a)
        b = self._findIndex(b)

        # avoid cycles by choosing the smallest label as the common one
        # swap such that a is smaller
        if a > b:
            a, b = b, a

        self._map[b] = a

    @threadsafe
    def makeNewIndex(self):
        newLabel = self._nextFree
        self._nextFree += 1
        self._map[newLabel] = newLabel
        return newLabel

    @threadsafe
    def findIndex(self, a):
        return self._findIndex(a)

    def _findIndex(self, a):
        while a != self._map[a]:
            a = self._map[a]
        return a

    def __str__(self):
        s = "<UnionFindArray>\n{}".format(self._map)

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['_lock']
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)


class InfiniteLabelIterator(object):

    def __init__(self, n, dtype=_LABEL_TYPE):
        if not np.issubdtype(dtype, np.integer):
            raise ValueError("Labels must have an integral type")
        self.dtype = dtype
        self.n = n

    def next(self):
        a = self.dtype(self.n)
        assert a < np.iinfo(self.dtype).max, "Label overflow."
        self.n += 1
        return a
