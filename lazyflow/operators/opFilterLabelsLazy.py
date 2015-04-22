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
#		   http://ilastik.org/license/
###############################################################################

import numpy as np
from threading import Lock

from collections import defaultdict
from itertools import ifilter

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.operators.opFilterLabels import OpLabelFilteringABC
from lazyflow.operators import OpReorderAxes
from lazyflow.rtype import SubRegion
from lazyflow.roi import determine_optimal_request_blockshape

import logging
logger = logging.getLogger(__name__)

from opLazyRegionGrowing import OpLazyRegionGrowing


class OpFilterLabelsLazy(OpLazyRegionGrowing):
    """
    Given a labeled volume, discard labels that have too few pixels.
    Zero is used as the background label, and filtering is done
    seperately in each time/channel slice. 
    This operator tries to compute as few chunks as possible for a
    given ROI with the same result as a global filter operator would
    give.
    """
    name = "OpFilterLabelsLazy"
    category = "generic"

    # always 3-dim
    ChunkShape = InputSlot(optional=True)

    # slots from filteringABC
    Input = InputSlot() 
    MinLabelSize = InputSlot(stype='int')
    MaxLabelSize = InputSlot(optional=True, stype='int')
    BinaryOut = InputSlot(optional=True, value=False, stype='bool')

    Output = OutputSlot()

    # internal slots
    _Input = OutputSlot()
    _Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpFilterLabelsLazy, self).__init__(*args, **kwargs)

        self.__allocateManagementDicts()

        # reordering operators - we want to handle txyzc inside this operator
        self._opIn = OpReorderAxes(parent=self)
        self._opIn.AxisOrder.setValue('txyzc')
        self._opIn.Input.connect(self.Input)
        self._Input.connect(self._opIn.Output)
        self._Input.notifyDirty(self.__propagateDirty)

        self._opOut = OpReorderAxes(parent=self)
        self._opOut.Input.connect(self._Output)
        self.Output.connect(self._opOut.Output)

    def shape(self):
        return self.Input.meta.shape

    def chunkShape(self):
        return getattr(self, "_chunkShape", None)

    def setupOutputs(self):

        self.__allocateManagementDicts()

        # determine chunk shape first, because the parent class needs it
        if self.ChunkShape.ready():
            chunkShape = (1,) + self.ChunkShape.value + (1,)
        else:
            chunkShape = self._automaticChunkShape()
        self._chunkShape = chunkShape
        super(OpFilterLabelsLazy, self).setupOutputs()

        self._Output.meta.assignFrom(self._Input.meta)

        if self.BinaryOut.ready() and self.BinaryOut.value:
            self.__binary = True
        else:
            self.__binary = False
        if self.MinLabelSize.ready():
            self.__minSize = self.MinLabelSize.value
        else:
            self.__minSize = 0
        if self.MaxLabelSize.ready():
            self.__maxSize = self.MaxLabelSize.value
        else:
            self.__maxSize = np.inf

        # go back to original order
        self._opOut.AxisOrder.setValue(self.Input.meta.getAxisKeys())

    def propagateDirty(self, slot, subindex, roi):
        if slot is self.Input:
            # dirty handling is done via callback from _Input
            pass
        else:
            # all output is invalidated
            self.__allocateManagementDicts()
            self._Output.setDirty(slice(None))

    def __propagateDirty(self, slot, roi):
        """
        catch propagateDirty at reordered input
        """
        start = np.asarray(roi.start, dtype=int)
        stop = np.asarray(roi.stop, dtype=int)

        # clear management structures first
        def inside(key, t_ind, c_ind):
            b = key[t_ind] >= start[0]
            b = b and key[t_ind] < stop[0]
            b = b and key[c_ind] >= start[4]
            b = b and key[c_ind] < stop[4]
            return b

        for d, t_ind, c_ind in self.__managementDicts:
            keys = d.keys()
            relevant = filter(lambda k: inside(k, t_ind, c_ind),
                              keys)
            for key in relevant:
                try:
                    del d[key]
                except KeyError:
                    # probably cleaned up already
                    pass

        start[1:4] = 0
        stop[1:4] = slot.meta.shape[1:4]
        roi = SubRegion(self._Output,
                        start=tuple(start), stop=tuple(stop))
        self._Output.setDirty(roi)

    def __allocateManagementDicts(self):
        # keep track of management dicts
        self.__managementDicts = []

        self.__labelCount = defaultdict(lambda: defaultdict(int))
        self.__managementDicts.append((self.__labelCount, 0, 1))
        self.__chunkLocks = defaultdict(Lock)
        self.__managementDicts.append((self.__chunkLocks, 0, 4))
        self.__handled = defaultdict(bool)
        self.__managementDicts.append((self.__handled, 0, 4))
        # keep track of merged regions
        self.__mergeMap = defaultdict(list)
        self.__managementDicts.append((self.__mergeMap, 0, 4))

    def handleSingleChunk(self, chunk):
        with self.__chunkLocks[chunk]:
            if self.__handled[chunk]:
                return set()
            self.__handled[chunk] = True
            roi = self.chunkIndexToRoi(chunk)
            data = self.Input.get(roi).wait()
            try:
                bins = np.bincount(data.ravel())
            except TypeError:
                # On 32-bit systems, must explicitly convert from uint32 to int
                # (This fix is just for VM testing.)
                bins = np.bincount(data.astype(np.int).ravel())
            for i in range(1, len(bins)):
                self.__labelCount[(chunk[0], chunk[4])][i] += bins[i]
            f = lambda i: bins[i] > 0 and i > 0
            return set(ifilter(f, xrange(1, len(bins))))

    def mergeChunks(self, chunkA, chunkB):
        a, b = self.orderPair(chunkA, chunkB)
        reordered = a != chunkA
        
        with self.__chunkLocks[a]:
            if b in self.__mergeMap[a]:
                return set(), set()
            self.__mergeMap[chunkA].append(chunkB)

            hyperplane_roi_a, hyperplane_roi_b = \
                self.chunkIndexToHyperplane(a, b)
            hyperplane_index_a = hyperplane_roi_a.toSlice()
            hyperplane_index_b = hyperplane_roi_b.toSlice()

            hyperplane_a = self.Input.get(hyperplane_roi_a).wait()
            hyperplane_b = self.Input.get(hyperplane_roi_b).wait()
            inds = hyperplane_a == hyperplane_b

            workA = set(hyperplane_a[inds]) - set([0])
            workB = set(hyperplane_b[inds]) - set([0])
            if reordered:
                return workB, workA
            else:
                return workA, workB

    def fillResult(self, roi, result):
        req = self.Input.get(roi)
        req.writeInto(result)
        req.block()
        for t in range(roi.start[0], roi.stop[0]):
            for c in range(roi.start[4], roi.stop[4]):
                lc = self.__labelCount[(t, c)]
                if len(lc) == 0:
                    continue
                maxLabel = max(lc.keys())
                mapping = np.arange(maxLabel+1)
                if self.__binary:
                    mapping[mapping>0] = 1
                for label in lc:
                    n = lc[label]
                    if n > self.__maxSize or n < self.__minSize:
                        mapping[label] = 0
                result[t, ..., c] = mapping[result[t, ..., c]]

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

        chunkShape = determine_optimal_request_blockshape(
            max_shape, ideal_shape, ram_per_pixel, 1, ram)
        return chunkShape
