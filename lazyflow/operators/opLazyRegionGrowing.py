
from abc import abstractmethod

import numpy as np
import vigra

from threading import Lock, Condition
from collections import defaultdict
import itertools

import logging
logger = logging.getLogger(__name__)

from lazyflow.operator import Operator
from lazyflow.rtype import SubRegion
from opLazyConnectedComponents import threadsafe

# This class manages parallel executions of lazy algorithms, such that
# no region is computed more than once. You may need to subclass this
# manager.
class LazyManager(object):

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
            inters = work & otherWork
            if len(inters) > 0:
                work = work - inters
                others.add(otherProcess)
        if len(work) > 0:
            thisChunk.append((ticket, work))
        return work, others

    @threadsafe
    def mergeWork(self, a, b):
        return a | b


class OpLazyRegionGrowing(Operator):
    name = "OpLazyRegionGrowing"

    ########################### API ###########################

    # the manager property holds a LazyManager instance
    # feel free to provide your own subclass
    @property
    def manager(self):
        return self.__manager

    @manager.setter
    def manager(self, mgr):
        assert isinstance(mgr, LazyManager), "Invalid manager type"
        self.__manager = mgr

    #TODO specify when lock use is appropriate
    @property
    def lock(self):
        return self._lock

    # TODO how to use mergeMap?
    @property
    def mergeMap(self):
        return self.__mergeMap

    ############# Methods you *should* override #############

    @abstractmethod
    #TODO document signature work=hsc(ind)
    def handleSingleChunk(self, chunkIndex):
        raise NotImplementedError()

    @abstractmethod
    #TODO document signature w1, w2=hsc(c1, c2)
    def mergeChunks(self, chunkA, chunkB):
        raise NotImplementedError()

    # this one is called in setupOutputs and must return a 5-tuple
    @abstractmethod
    def chunkShape(self):
        raise NotImplementedError()

    # this one is called in setupOutputs and must return a 5-tuple
    @abstractmethod
    def shape(self):
        raise NotImplementedError()

    # final mapping of result, after region is grown
    @abstractmethod
    def fillResult(self):
        raise NotImplementedError()

    def execute(self, slot, subindex, roi, result):
        self.executeRegionGrowing(roi, result)

    def propagateDirty(self, slot, subindex, roi):
        pass

    def setInSlot(self, slot, subindex, key, value):
        pass

    ############# Methods you *can* override #############
    ############# (but please call super()!) #############

    def __init__(self, *args, **kwargs):
        super(OpLazyRegionGrowing, self).__init__(*args, **kwargs)
        self._lock = Lock()

    def setupOutputs(self):
        self.__setDefaultInternals()

    ############# Methods you can call #############

    def executeRegionGrowing(self, roi, result):
        logger.debug("Growing region for {}".format(roi))
        self.__manager.hello()
        othersToWaitFor = set()
        chunks = self.roiToChunkIndex(roi)
        for chunk in chunks:
            othersToWaitFor |= self.growRegion(chunk)

        self.__manager.waitFor(othersToWaitFor)
        self.__manager.goodbye()
        self.fillResult(roi, result)

    # create roi object from chunk index
    def chunkIndexToRoi(self, index):
        shape = self.__shape
        start = self.__chunkShape * np.asarray(index)
        stop = self.__chunkShape * (np.asarray(index) + 1)
        stop = np.where(stop > shape, shape, stop)
        roi = SubRegion(self.Input,
                        start=tuple(start), stop=tuple(stop))
        return roi

    # create a list of chunk indices needed for a particular roi
    def roiToChunkIndex(self, roi):
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

    # compute the adjacent hyperplanes of two chunks (1 pix wide)
    # @return 2-tuple of roi's for the respective chunk
    def chunkIndexToHyperplane(self, chunkA, chunkB, width=2):
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

    # generate a list of adjacent chunks
    def generateNeighbours(self, chunkIndex):
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

    # order a pair of chunk indices lexicographically
    # (ret[0] is top-left-in-front-of of ret[1])
    @staticmethod
    def orderPair(tupA, tupB):
        for a, b in zip(tupA, tupB):
            if a < b:
                return tupA, tupB
            if a > b:
                return tupB, tupA
        raise ValueError("tupA={} and tupB={} are the same".format(tupA, tupB))
        return tupA, tupB

    ################## INTERNALS -- DO NOT USE ##################

    # grow the requested region such that all chunks inside that region
    # are final
    # @param chunkIndex the index of the chunk to finalize
    def growRegion(self, chunkIndex):
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
                otherWork = self.mergeChunks(currentChunk, other)

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

    # fills attributes with standard values, call on each setupOutputs
    def __setDefaultInternals(self):
        shape = self.shape()
        chunkShape = self.chunkShape()
        chunkShape = np.minimum(shape, chunkShape)
        f = lambda i: shape[i]//chunkShape[i]\
            + (1 if shape[i] % chunkShape[i] else 0)
        self.__chunkArrayShape = tuple(map(f, range(len(shape))))
        self.__chunkShape = np.asarray(chunkShape, dtype=np.int)
        self.__shape = shape
        # manager object
        self.__manager = LazyManager()

        ### algorithmic ###

        # keep track of merged regions
        self.__mergeMap = defaultdict(list)
