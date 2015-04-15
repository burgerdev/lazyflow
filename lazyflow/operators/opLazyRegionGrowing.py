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


from abc import abstractmethod

import numpy as np
import vigra

from threading import Lock, Condition
from collections import defaultdict
import itertools
import functools

import logging
logger = logging.getLogger(__name__)

from lazyflow.operator import Operator
from lazyflow.rtype import SubRegion


def threadsafe(method):
    """
    decorator locking a method for the whole program

    Needs the property self._lock to expose the Lock interface.
    """
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


class LazyManager(object):
    """
    This class manages the parallel execution of lazy algorithms, such
    that no region is computed more than once. The manager handles work
    packages which must implement the set() interface, or at least
        * difference_update
        * union
        * intersection
        * len
    The workflow for using the manager looks something like this

        >>> mgr = LazyManager()
        >>> mgr.hello()
        >>> myTicket = mgr.register()
        >>> # determine my work package for my chunk
        >>> # myChunk = ...
        >>> # myWork = ...
        >>> myRemainingWork, othersToWaitFor = \
        ... mgr.checkoutWork(myChunk, myWork, myTicket)
        >>> # do work ...
        >>> mgr.unregister(myTicket)
        >>> # now wait until all other workers have finished their work
        >>> # on this chunk
        >>> mgr.waitFor(othersToWaitFor)
        >>> # all work on myChunk involving myWork is done by now
        >>> mgr.goodbye()
    """

    def __init__(self):
        self._lock = Condition()
        self._registered = set()
        self._total = 0
        self._asleep = 0
        self._iterator = itertools.count()
        self._work = defaultdict(list)

    # Call this before interacting with the manager object. Just used
    # for detecting cyclic wait() calls now.
    @threadsafe
    def hello(self):
        self._total += 1

    # Call at the end of execute()
    @threadsafe
    def goodbye(self):
        self._total -= 1

    # Call this if you are going to take responsibility for some chunks
    @threadsafe
    def register(self):
        n = self._iterator.next()
        self._registered.add(n)
        return n

    # Call when done with all registered chunks.
    @threadsafe
    def unregister(self, n):
        self._registered.remove(n)
        self._lock.notify_all()

    # call to wait for other processes
    @threadsafe
    def waitFor(self, others):
        others = set(others)
        remaining = others & self._registered
        while len(remaining) > 0:
            self._asleep += 1
            if self._asleep == self._total:
                raise RuntimeError("Cyclic waiting detected")
            self._lock.wait()
            self._asleep -= 1
            remaining &= self._registered

    @threadsafe
    def checkoutWork(self, chunk, work, ticket):
        others = set()
        thisChunk = self._work[chunk]
        for otherProcess, otherWork in thisChunk:
            inters = work.intersection(otherWork)
            if len(inters) > 0:
                work.difference_update(inters)
                others.add(otherProcess)
        if len(work) > 0:
            thisChunk.append((ticket, work))
        return work, others

    @threadsafe
    def mergeWork(self, a, b):
        return a.union(b)


class OpLazyRegionGrowing(Operator):
    """
    provides a framework for making operations lazy by growing the roi

    
    """
    name = "OpLazyRegionGrowing"

    ########################### API ###########################

    @property
    def manager(self):
        """
        a LazyManager instance suitable for the task
        """
        return self.__manager

    @manager.setter
    def manager(self, mgr):
        assert isinstance(mgr, LazyManager), "Invalid manager type"
        self.__manager = mgr

    ############# Methods you *should* override #############

    @abstractmethod
    def handleSingleChunk(self, chunkIndex):
        """
        process  a single chunk

        This method usually applies non-lazy implementations to a single
        chunk. Treatment of borders is done in mergeChunks(). The return
        value must be a work package that can be handled by the manager,
        e.g. a dict for the default LazyManager. Make sure that you use
        locks for chunks inside this function if you need them, this
        operator will not lock per-chunk.
        """
        raise NotImplementedError()

    @abstractmethod
    def mergeChunks(self, chunkA, chunkB):
        """
        border treatment of two adjacent chunks

        This method is called after handleSingleChunk was called for
        both chunkA and chunkB, to unify the output of underlying non-
        lazy implementations. The return value must be a 2-tuple of work
        packages, the first one for chunkA and the second one for
        chunkB.

        This method should probably be protected by a lock for a
        specified chunk, e.g. the lexicographically smaller one.
        Careful: There is no fixed ordering of the arguments you
        probably have to reorder them before use (see method orderPair).

        It is *not* guaranteed that this method will be called only once
        for a particular pair of chunks.
        """
        raise NotImplementedError()

    @abstractmethod
    def chunkShape(self):
        """
        a tuple with length equal to the input's dimension

        Determining the chunk shape is left to child classes, so that
        they can use a slot, meta.ideal_blockshape, ...
        """
        raise NotImplementedError()

    # this one is called in setupOutputs and must return a 5-tuple
    @abstractmethod
    def shape(self):
        """
        a tuple with length equal to the input's dimension

        Determining the shape is left to child classes, so that
        arbitrary input slots can be supported.
        """
        raise NotImplementedError()

    @abstractmethod
    def fillResult(self, roi, result):
        """
        final mapping of result, after region is grown
        
        This method must fill up the result with the data computed by
        the region growing, probably from an internal cache.
        """
        raise NotImplementedError()

    def execute(self, slot, subindex, roi, result):
        self.executeRegionGrowing(roi, result)

    # The methods below belong to the operator interface, we don't want
    # to change the default behaviour by defining them here. In child
    # operators they should be implemented, though.
    #
    # def propagateDirty(self, slot, subindex, roi):
    #     pass

    # def setInSlot(self, slot, subindex, key, value):
    #     pass

    ############# Methods you *can* override #############
    ############# (but please call super()!) #############

    def __init__(self, *args, **kwargs):
        super(OpLazyRegionGrowing, self).__init__(*args, **kwargs)
        self._lock = Lock()

    def setupOutputs(self):
        self.__setDefaultInternals()

    ############# Methods you can call #############

    def executeRegionGrowing(self, roi, result):
        """
        start the region growing algorithm
        
        Note that this operator is slot agnostic, meaning that it
        expects requests to come for exactly one slot. This can be
        changed by implementing execute().
        """
        logger.debug("Growing region for {}".format(roi))
        self.__manager.hello()
        othersToWaitFor = set()
        chunks = self.roiToChunkIndex(roi)
        for chunk in chunks:
            othersToWaitFor |= self.__growRegion(chunk)

        self.__manager.waitFor(othersToWaitFor)
        self.__manager.goodbye()
        self.fillResult(roi, result)

    def chunkIndexToRoi(self, index):
        """
        create roi object from chunk index
        """
        shape = self.__shape
        start = self.__chunkShape * np.asarray(index)
        stop = self.__chunkShape * (np.asarray(index) + 1)
        stop = np.where(stop > shape, shape, stop)
        roi = SubRegion(self.Input,
                        start=tuple(start), stop=tuple(stop))
        return roi

    def roiToChunkIndex(self, roi):
        """
        create a list of chunk indices needed for a particular roi
        """
        cs = self.__chunkShape
        start = np.asarray(roi.start)
        stop = np.asarray(roi.stop)
        start_cs = start / cs
        stop_cs = stop / cs
        # add one if division was not even
        stop_cs += np.where(stop % cs, 1, 0)
        iters = [xrange(start_cs[i], stop_cs[i]) for i in range(5)]
        chunks = list(itertools.product(*iters))
        return chunks

    def chunkIndexToHyperplane(self, chunkA, chunkB, width=2):
        """
        compute the adjacent hyperplanes of two chunks (1 pix wide)

        @return 2-tuple of roi's for the respective chunk
        """
        rev = False
        assert chunkA[0] == chunkB[0] and chunkA[4] == chunkB[4],\
            "these chunks are not spatially adjacent"

        # just iterate over spatial axes
        for i in range(1, 4):
            if chunkA[i] > chunkB[i]:
                rev = True
                chunkA, chunkB = chunkB, chunkA
            if chunkA[i] < chunkB[i]:
                roiA = self.chunkIndexToRoi(chunkA)
                roiB = self.chunkIndexToRoi(chunkB)
                start = np.asarray(roiA.start)
                start[i] = roiA.stop[i] - width//2
                roiA.start = tuple(start)
                stop = np.asarray(roiB.stop)
                stop[i] = roiB.start[i] + width//2
                roiB.stop = tuple(stop)
        if rev:
            return roiB, roiA
        else:
            return roiA, roiB

    def generateNeighbours(self, chunkIndex):
        """
        generate a list of adjacent chunks
        """
        n = []
        idx = np.asarray(chunkIndex, dtype=np.int)
        # only spatial neighbours are considered
        for i in range(1, 4):
            if idx[i] > 0:
                new = idx.copy()
                new[i] -= 1
                n.append(tuple(new))
            if idx[i]+1 < self.__chunkArrayShape[i]:
                new = idx.copy()
                new[i] += 1
                n.append(tuple(new))
        return n

    @staticmethod
    def orderPair(tupA, tupB):
        """
        order a pair of chunk indices lexicographically
        (ret[0] is top-left-in-front-of of ret[1])
        """
        for a, b in zip(tupA, tupB):
            if a < b:
                return tupA, tupB
            if a > b:
                return tupB, tupA
        raise ValueError("tupA={} and tupB={} are the same".format(tupA, tupB))
        return tupA, tupB

    ################## INTERNALS -- DO NOT USE ##################

    def __growRegion(self, chunkIndex):
        """
        grow the requested region until all chunks inside are final

        @param chunkIndex the index of the chunk to finalize
        """
        ticket = self.__manager.register()
        othersToWaitFor = set()

        # handle this chunk
        workLoad = self.handleSingleChunk(chunkIndex)
        chunksToProcess = [(chunkIndex, workLoad)]

        while chunksToProcess:
            # Breadth-First-Search, using list as FIFO
            currentChunk, currentWorkload = chunksToProcess.pop(0)

            # tell the manager that we are about to finalize something
            actualWorkload, others = \
                self.__manager.checkoutWork(currentChunk,
                                            currentWorkload,
                                            ticket)
            othersToWaitFor |= others

            # start processing our own piece of work

            # start merging adjacent regions
            otherChunks = self.generateNeighbours(currentChunk)
            for other in otherChunks:
                self.handleSingleChunk(other)
                _, otherWork = self.mergeChunks(currentChunk, other)

                # add the neighbour to our processing queue only if it actually
                # shares objects
                if len(otherWork) > 0:
                    # check if already in queue
                    found = False
                    for i in xrange(len(chunksToProcess)):
                        if chunksToProcess[i][0] == other:
                            otherWork2 = chunksToProcess[i][1]
                            work = self.manager.mergeWork(otherWork,
                                                          otherWork2)
                            chunksToProcess[i] = (other, work)
                            found = True
                            break
                    if not found:
                        chunksToProcess.append((other, otherWork))

        self.__manager.unregister(ticket)
        return othersToWaitFor

    def __setDefaultInternals(self):
        """
        fill attributes with standard values (call on each setupOutputs)
        """
        shape = self.shape()
        chunkShape = self.chunkShape()
        assert len(shape) == len(chunkShape)
        chunkShape = np.minimum(shape, chunkShape)
        f = lambda i: shape[i]//chunkShape[i]\
            + (1 if shape[i] % chunkShape[i] else 0)
        self.__chunkArrayShape = tuple(map(f, range(len(shape))))
        self.__chunkShape = np.asarray(chunkShape, dtype=np.int)
        self.__shape = shape
        # manager object
        self.__manager = LazyManager()
