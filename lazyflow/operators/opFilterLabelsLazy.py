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

import logging
logger = logging.getLogger(__name__)

from opLazyRegionGrowing import OpLazyRegionGrowing
from opLazyRegionGrowing import LazyManager
from opLazyRegionGrowing import threadsafe


class FilterManager(LazyManager):
    def __init__(self):
        self._work = defaultdict(list)

    @threadsafe
    def checkoutWork(self, chunk, work, ticket):
        others = set()
        thisChunk = self._work[chunkIndex]
        for otherProcess, otherWork in thisChunk:
            inters = work & otherWork
            if len(inters) > 0:
                work = work - inters
                others.add(otherProcess)
        if len(work) > 0:
            thisChunk.append((n, work))
        return work, others


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
        return self.ChunkShape.value

    def setupOutputs(self):
        super(OpFilterLabelsLazy, self).setupOutputs()

        self.__labelCount = defaultdict(int)
        self.__chunkLocks = defaultdict(Lock)
        self.__handled = set()

        self.Output.meta.assignFrom(self.Input.meta)

        if self.BinaryOut.ready() and self.BinaryOut.value:
            self.Output.meta.dtype = np.uint8
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
                self.__labelCount[i] += bins[i]
            f = lambda i: bins[i] > 0 and i > 0
            return set(ifilter(f, xrange(1, len(bins))))

    def mergeChunks(self, chunkA, chunkB):
        a, b = self.orderPair(chunkA, chunkB)
        reordered = a != chunkA
        
        with self.__chunkLocks[a]:
            if b in self.mergeMap[a]:
                return set()
            self.mergeMap[chunkA].append(chunkB)

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
        lc = self.__labelCount
        maxLabel = max(lc.keys())
        mapping = np.arange(maxLabel+1)
        for label in lc:
            c = lc[label]
            if c > self.__maxSize or c < self.__minSize:
                mapping[label] = 0
        req = self.Input.get(roi)
        req.writeInto(result)
        req.block()
        result[:] = mapping[result]
