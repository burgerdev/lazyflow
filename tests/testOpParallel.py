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

import unittest

import numpy as np
import vigra

from lazyflow .operator import Operator, InputSlot, OutputSlot
from lazyflow.graph import Graph
from lazyflow.rtype import SubRegion

from lazyflow.operators.opParallel import OpParallel

from multiprocessing import Process, Queue


class OpToTest(Operator):
    Input = InputSlot()
    Output = OutputSlot()
    nExecutes = 0
    nDirty = 0

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)

    def execute(self, slot, subindex, roi, result):
        result[:] = self.Input.get(roi).wait() + 1

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi)


def chunk(start, stop, step):
    if len(start) == 0:
        return [(tuple(), tuple())]
    chunks = []
    subs = chunk(start[1:], stop[1:], step[1:])
    for i in range(start[0], stop[0], step[0]):
        for c in subs:
            newStart = (i,) + c[0]
            newStop = (min(i+step[0], stop[0]),) + c[1]
            chunks.append((newStart, newStop))
    return chunks


def roiParallelizer(meta, roi, result):
    start = roi.start
    resstart = np.asarray(start)
    stop = roi.stop
    chunks = chunk(start, stop, tuple(s//2 for s in stop))
    portions = []
    resultPortions = []
    for start, stop in chunks:
        a = np.asarray(start)
        b = np.asarray(stop)
        portions.append(SubRegion(None, start=start, stop=stop))
        b -= resstart
        a -= resstart
        resultPortions.append(SubRegion(None, start=tuple(a), stop=tuple(b)))

    return portions, resultPortions


class OpReq(OpParallel):
    def assignWork(self, portions, result):
        reqs = [self._op.outputs[self.s2p].get(roi) for roi in portions[0]]
        for req, roi in zip(reqs, portions[1]):
            req.writeInto(result[roi.toSlice()])
            req.submit()
        [req.wait() for req in reqs]


class OpMultiProc(OpParallel):
    def assignWork(self, portions, result):
        def fun(roi, resroi, queue):
            req = self._op.outputs[self.s2p].get(roi)
            res = req.wait()
            queue.put((resroi, res))
            queue.close()

        queue = Queue()
        procs = [Process(target=fun, args=(roi, resroi, queue))
                 for roi, resroi in zip(portions[0], portions[1])]
        [p.start() for p in procs]
        n = len(procs)
        while n > 0:
            roi, res = queue.get()
            result[roi.toSlice()] = res
            n -= 1
        queue.close()
        [p.join() for p in procs]


class TestOpParallel(unittest.TestCase):
    def setUp(self):
        vol = np.zeros((100, 100, 100), dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')
        self.vol = vol
        res = np.ones_like(vol)
        res = vigra.taggedView(res, axistags=vol.axistags)
        self.res = res

    def testRequest(self):

        op = OpReq(OpToTest, "Output", graph=Graph())
        op.Input.setValue(self.vol)
        op.RoiParallelizer.setValue(roiParallelizer)

        res = op.Output[...].wait()
        res = vigra.taggedView(res, axistags=op.Output.meta.axistags)
        np.testing.assert_array_equal(res, self.res)

        sl = SubRegion(None, start=(10, 20, 30), stop=(41, 42, 43)).toSlice()
        res = op.Output[sl].wait()
        res = vigra.taggedView(res, axistags=op.Output.meta.axistags)
        np.testing.assert_array_equal(res, self.res[sl])

    def testMultiProcessing(self):

        op = OpMultiProc(OpToTest, "Output", graph=Graph())
        op.Input.setValue(self.vol)
        op.RoiParallelizer.setValue(roiParallelizer)

        res = op.Output[...].wait()
        res = vigra.taggedView(res, axistags=op.Output.meta.axistags)
        np.testing.assert_array_equal(res, self.res)

        sl = SubRegion(None, start=(10, 20, 30), stop=(41, 42, 43)).toSlice()
        res = op.Output[sl].wait()
        res = vigra.taggedView(res, axistags=op.Output.meta.axistags)
        np.testing.assert_array_equal(res, self.res[sl])


if __name__ == "__main__":

    vol = np.zeros((100, 100, 100), dtype=np.uint8)
    vol = vigra.taggedView(vol, axistags='xyz')

    op = OpMultiProc(OpToTest, "Output", graph=Graph())
    op.Input.setValue(vol)
    op.RoiParallelizer.setValue(roiParallelizer)

    res = op.Output[...].wait()
