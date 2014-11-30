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
    blockShape = np.asarray(blockShape, dtype=np.int)
    blocks = getIntersectingBlocks(blockShape, (roi.start, roi.stop))

    def b2r(block):
        a = np.maximum(roi.start, block)
        b = np.minimum(roi.stop, block + blockShape)
        return SubRegion(roi.slot, start=a, stop=b)

    return map(b2r, blocks)


def mortonOrderIterator(shape):
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
    a = np.asarray(roi.start, dtype=np.int)
    b = np.asarray(roi.stop, dtype=np.int)
    o = np.asarray(origin, dtype=np.int)
    b -= o
    a -= o
    return SubRegion(roi.slot, start=tuple(a), stop=tuple(b))


class RequestStrategy(ParallelStrategyABC):
    """
    TODO doc
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
            req.writeInto(result[r.toSlice()])
            req.submit()
        for req, r in reqs:
            req.block()


# see http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class
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
    """

    def __init__(self, nWorkers, blockShape):
        self._nWorkers = nWorkers
        self._blockShape = blockShape

    def map(self, slot, roi, result):
        """
        partitions the ROI into blocks, starts request for each block
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
            result[roi] = res
            n -= 1
        queue.close()
        [p.join() for p in procs]
