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

from lazyflow.graph import Graph
from lazyflow.rtype import SubRegion
from lazyflow.operator import Operator, InputSlot, OutputSlot

from lazyflow.operators.opParallel import OpMapParallel
from lazyflow.operators.opParallelImplementations import RequestStrategy
from lazyflow.operators.opParallelImplementations import MultiprocessingStrategy

from lazyflow.operators.opParallelImplementations import partitionRoi
from lazyflow.operators.opParallelImplementations import partitionRoiForWorkers
from lazyflow.operators.opParallelImplementations import toResultRoi
from lazyflow.operators.opParallelImplementations import mortonOrderIterator


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


class TestFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def testPartitionRoi(self):
        vol = np.zeros((30, 30, 30), dtype=np.uint8)
        blockShape = (10, 12, 14)
        start = (2, 12, 7)
        stop = (25, 13, 20)
        roi = SubRegion(None, start=start, stop=stop)
        exp = vol.copy()
        exp[roi.toSlice()] = 1

        rois = partitionRoi(roi, blockShape)
        for r in rois:
            vol[r.toSlice()] += 1
        np.testing.assert_array_equal(vol, exp)

    def testMortonOrder(self):
        mortonOrder = lambda x: list(mortonOrderIterator(x))
        order = mortonOrder((4, 4))
        assert len(order) == 16
        for idx in order:
            assert len(idx) == 2
        order = mortonOrder((2, 2, 2))
        assert len(order) == 8
        for idx in order:
            assert len(idx) == 3
        # FIXME no real test here

    def testPartitionRoiForWorkers(self):
        vol = np.zeros((20, 20, 20), dtype=np.uint8)
        blockShape = (5, 5, 5)
        start = (0, 0, 0)
        stop = (20, 20, 20)
        roi = SubRegion(None, start=start, stop=stop)
        exp = vol.copy()
        exp[roi.toSlice()] = 1

        nWorkers = 8
        workerRois = partitionRoiForWorkers(roi, vol.shape,
                                            blockShape, nWorkers)
        assert len(workerRois) == nWorkers, "did not split correctly"
        for i, rois in enumerate(workerRois, start=1):
            for r in rois:
                vol[r.toSlice()] += i
        assert vol[0:10, 0:10, 0:10].var() < 1e-10
        assert vol[10:20, 0:10, 0:10].var() < 1e-10
        assert vol[0:10, 10:20, 0:10].var() < 1e-10
        assert vol[10:20, 10:20, 0:10].var() < 1e-10

    def testToResultRoi(self):
        start = (5, 6, 7)
        stop = (6, 7, 8)
        roi = SubRegion(None, start=start, stop=stop)
        newroi = toResultRoi(roi, (3, 2, 1))
        np.testing.assert_array_equal(newroi.start, (2, 4, 6))
        np.testing.assert_array_equal(newroi.stop, (3, 5, 7))


class TestStrategies(unittest.TestCase):
    def setUp(self):
        vol = np.zeros((100, 100, 100), dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')
        self.vol = vol
        res = np.ones_like(vol)
        res = vigra.taggedView(res, axistags=vol.axistags)
        self.res = res

    def testRequestStrategy(self):
        strat = RequestStrategy((30, 25, 45))
        op = OpMapParallel(OpToTest, "Output", strat, graph=Graph())
        op.Input.setValue(self.vol)

        res = op.Output[...].wait()
        res = vigra.taggedView(res, axistags=op.Output.meta.axistags)
        np.testing.assert_array_equal(res, self.res)

        sl = SubRegion(None, start=(10, 20, 30), stop=(41, 42, 43)).toSlice()
        res = op.Output[sl].wait()
        res = vigra.taggedView(res, axistags=op.Output.meta.axistags)
        np.testing.assert_array_equal(res, self.res[sl])


    def testMultiprocessingStrategy(self):
        strat = MultiprocessingStrategy(4, (30, 25, 45))
        op = OpMapParallel(OpToTest, "Output", strat, graph=Graph())
        op.Input.setValue(self.vol)

        res = op.Output[...].wait()
        res = vigra.taggedView(res, axistags=op.Output.meta.axistags)
        x, y, z = np.where(res!=self.res)
        np.testing.assert_array_equal(res, self.res)

        sl = SubRegion(None, start=(10, 20, 30), stop=(41, 42, 43)).toSlice()
        res = op.Output[sl].wait()
        res = vigra.taggedView(res, axistags=op.Output.meta.axistags)
        np.testing.assert_array_equal(res, self.res[sl])
