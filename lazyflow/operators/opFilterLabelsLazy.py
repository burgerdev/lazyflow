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
from lazyflow.operators.opLazyConnectedComponents import OpLazyConnectedComponents

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

    def shape(self):
        return self.Input.meta.shape

    def chunkShape(self):
        return getattr(self, "_chunkShape", None)

    def setupOutputs(self):

        # determine chunk shape first, because the parent class needs it
        if self.ChunkShape.ready():
            chunkShape = (1,) + self.ChunkShape.value + (1,)
        elif self._Input.meta.ideal_blockshape is not None and\
                np.prod(self._Input.meta.ideal_blockshape) > 0:
            chunkShape = self._Input.meta.ideal_blockshape
        else:
            chunkShape = OpLazyConnectedComponents._automaticChunkShape(
                self._Input.meta.shape)
        self._chunkShape = chunkShape
        super(OpFilterLabelsLazy, self).setupOutputs()

        self.__labelCount = defaultdict(lambda: defaultdict(int))
        self.__chunkLocks = defaultdict(Lock)
        self.__handled = set()
        # keep track of merged regions
        self.__mergeMap = defaultdict(list)

        self.Output.meta.assignFrom(self.Input.meta)

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

    def propagateDirty(self, slot, subindex, roi):
        # set everything dirty because determining what changed is as
        # expensive as recomputing
        self.Output.setDirty(slice(None))

    def handleSingleChunk(self, chunk):
        with self.__chunkLocks[chunk]:
            if chunk in self.__handled:
                return set()
            self.__handled.add(chunk)
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
                return set()
            self.__mergeMap[chunkA].append(chunkB)

            hyperplane_roi_a, hyperplane_roi_b = \
                self.chunkIndexToHyperplane(a, b)
            hyperplane_index_a = hyperplane_roi_a.toSlice()
            hyperplane_index_b = hyperplane_roi_b.toSlice()

            hyperplane_a = self.Input.get(hyperplane_roi_a).wait()
            hyperplane_b = self.Input.get(hyperplane_roi_b).wait()
            inds = hyperplane_a == hyperplane_b

            if reordered:
                work = set(hyperplane_a[inds]) - set([0])
            else:
                work = set(hyperplane_b[inds]) - set([0])
            return work
            

    def fillResult(self, roi, result):
        req = self.Input.get(roi)
        req.writeInto(result)
        req.block()
        for t in range(roi.start[0], roi.stop[0]):
            for c in range(roi.start[4], roi.stop[4]):
                lc = self.__labelCount[(t, c)]
                maxLabel = max(lc.keys())
                mapping = np.arange(maxLabel+1)
                if self.__binary:
                    mapping[mapping>0] = 1
                for label in lc:
                    n = lc[label]
                    if n > self.__maxSize or n < self.__minSize:
                        mapping[label] = 0
                result[t, ..., c] = mapping[result[t, ..., c]]
