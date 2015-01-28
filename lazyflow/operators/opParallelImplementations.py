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

from opParallel import ParallelStrategyABC
from lazyflow.roi import getIntersectingBlocks, getIntersection
from lazyflow.rtype import SubRegion

from multiprocessing import Process, Queue
from itertools import izip, repeat, imap, chain, ifilter, groupby, count
from functools import partial


def partitionRoi(roi, blockShape):
    '''
    straightforward partitioning function
    @input roi the ROI which shall be partitioned
    @param blockShape the shape of a single partition
    @return list of SubRegion objects corresponding to partitions
    '''
    blockShape = np.asarray(blockShape, dtype=np.int)
    blocks = getIntersectingBlocks(blockShape, (roi.start, roi.stop))

    def b2r(block):
        a = np.maximum(roi.start, block)
        b = np.minimum(roi.stop, block + blockShape)
        return SubRegion(roi.slot, start=a, stop=b)

    return map(b2r, blocks)


def mortonOrderIterator(shape):
    '''
    @param shape the array shape over which to iterate
    @return iterator over single point indices in Morton order (Z-order)
    '''
    n = len(shape)
    x = np.zeros((n,), dtype=np.int)
    lim = np.prod(shape)
    counter = count(0)
    returned = 0
    while returned < lim:
        k = 0
        c = counter.next()
        x *= 0
        while c > 0:
            for i in xrange(n):
                x[i] |= c % 2 << k
                c //= 2
            k += 1
        if np.all(x < shape):
            yield tuple(x)
            returned += 1


def partitionRoiForWorkers(roi, shape, blockShape, n):
    '''
    Partition a ROI and distribute it for n workers. The entire volume is
    distributed over n workers and *after that* the roi is intersected with
    the sets for each worker. This means that a ROI will always be assigned to
    the same worker, regardless of the remaining request.
    
    @param roi roi which shall be distributed
    @param shape shape of the entire volume
    @param blockShape shape of a single block
    @paran n number of workers
    @return a n-item list of lists of SubRegion objects
    '''
    blockShape = np.asarray(blockShape, dtype=np.int)
    blocks = getIntersectingBlocks(blockShape, ((0,)*len(shape), shape),
                                   asarray=True)
    numBlocks = int(np.prod(blocks.shape[:-1]))
    morton = mortonOrderIterator(blocks.shape[:-1])
    blocksPerWorker = numBlocks // n
    if blocksPerWorker * n < numBlocks:
        blocksPerWorker += 1
    workers = chain(*[repeat(i, blocksPerWorker) for i in range(n)])
    mortonWithWorker = izip(morton, workers)
    mortonStarts = imap(lambda x: (blocks[x[0]], x[1]), mortonWithWorker)
    mortonRois = imap(lambda x: ((x[0], x[0] + blockShape), x[1]),
                      mortonStarts)
    fn = partial(getIntersection, (roi.start, roi.stop), assertIntersect=False)
    mortonRois = imap(lambda x: (fn(x[0]), x[1]), mortonRois)
    mortonRoisFiltered = ifilter(lambda x: x[0] is not None, mortonRois)
    mortonSubRegions = imap(
        lambda x: (SubRegion(roi.slot, x[0][0], x[0][1]), x[1]),
        mortonRoisFiltered)

    mortonGrouped = groupby(mortonSubRegions, key=lambda x: x[1])
    res = [()]*n
    for k, g in mortonGrouped:
        res[k] = tuple(map(lambda x: x[0], g))
    return res


def toResultRoi(roi, origin):
    '''
    transform an input ROI so that it can be used as result roi (in execute())
    Example:
    
    def execute(self, slot, subindex, roi, result):
        roi2 = shrinkRoi(roi)
        data = self.Input.get(roi2)
        result_roi = toResultRoi(roi2, roi.start)
        result[result_roi.toSlice()] = data
    
    '''
    a = np.asarray(roi.start, dtype=np.int)
    b = np.asarray(roi.stop, dtype=np.int)
    o = np.asarray(origin, dtype=np.int)
    b -= o
    a -= o
    return SubRegion(roi.slot, start=tuple(a), stop=tuple(b))


class RequestStrategy(ParallelStrategyABC):
    """
    TODO doc
    FIXME not working for Opaque output slots that return a value
          (but do those exist, anyway?)
    """

    def __init__(self, blockShape):
        self._blockShape = blockShape

    def map(self, slot, roi, result):
        """
        partitions the ROI into blocks, starts request for each block
        """
        rois = partitionRoi(roi, self._blockShape)
        reqs = [(slot.get(r), toResultRoi(r, roi.start)) for r in rois]
        for req, r in reqs:
            if result is not None:
                req.writeInto(result[r.toSlice()])
            req.submit()
        for req, r in reqs:
            req.block()


# Using arbitrary functions as multiprocessing.Process targets is hard. We use
# some wrappers to ensure all functions can be pickled.
# see http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class
# TODO check whether this is still a problem
def spawn(f):
    def fun(*args, **kwargs):
        f(*args, **kwargs)
    return fun


def parmap(f, X):
    proc = [Process(target=spawn(f), args=(x))
            for x in X]
    return proc


class MultiprocessingStrategy(ParallelStrategyABC):
    """
    TODO doc
    FIXME not working for Opaque output slots that return a value
          (but do those exist, anyway?)
    """

    def __init__(self, nWorkers, blockShape):
        self._nWorkers = nWorkers
        self._blockShape = blockShape

    def map(self, slot, roi, result):
        """
        partitions the ROI into block sets for n workers, starts processes
        for each worker
        """
        workerRois = partitionRoiForWorkers(roi, slot.meta.shape,
                                            self._blockShape, self._nWorkers)

        origin = roi.start

        queue = Queue()

        def multiProcFun(rois, queue):
            for roi in rois:
                req = slot.get(roi)
                res = req.wait()
                resroi = toResultRoi(roi, origin)
                queue.put((resroi.toSlice(), res))
            queue.close()

        procs = parmap(multiProcFun, izip(workerRois, repeat(queue)))
        [p.start() for p in procs]
        n = np.sum([len(p) for p in workerRois])
        while n > 0:
            roi, res = queue.get()
            if result is not None:
                result[roi] = res
            n -= 1
        queue.close()
        [p.join() for p in procs]


try:
    from mpi4py import MPI
except ImportError:
    have_mpi = False
else:
    have_mpi = True


class MPIStrategy(ParallelStrategyABC):
    """
    TODO doc
    FIXME not working for Opaque output slots that return a value
          (but do those exist, anyway?)
    """

    def __init__(self, blockShape):
        assert have_mpi, "Could not load mpi4py"
        self._blockShape = blockShape

    def map(self, slot, roi, result):
        """
        use MPI ranks 1-n for computation, rank 0 for receiving messages
        and further computations
        """
        comm = MPI.COMM_WORLD
        total = comm.size
        assert total > 1, "Need more than 1 process to use MPIStrategy"
        me = comm.rank
        workerRois = partitionRoiForWorkers(roi, slot.meta.shape,
                                            self._blockShape,
                                            total - 1)

        if me == 0:
            numBlocks = sum(len(x) for x in workerRois)
            self._receive(result, numBlocks)
            return
        else:
            offset = sum(len(x) for x in workerRois[:me-2])
            myRois = workerRois[me - 1]

        origin = roi.start
        count = 0
        for roi in myRois:
            print("{}: {}".format(me, str(roi)))
            req = slot.get(roi)
            res = req.wait()
            resroi = toResultRoi(roi, origin).toSlice()
            # TODO use array slices and Ibsend/Recv
            comm.send((resroi, res), dest=0, tag=offset+count)
            count += 1

    def _receive(self, result, n):
        '''
        only the root process is receiving data
        '''
        comm = MPI.COMM_WORLD
        assert comm.rank == 0
        while n > 0:
            # TODO handle errors in workers
            sl, block = comm.recv(source=MPI.ANY_SOURCE,
                                  tag=MPI.ANY_TAG)
            if result is not None:
                result[sl] = block
            n -= 1
