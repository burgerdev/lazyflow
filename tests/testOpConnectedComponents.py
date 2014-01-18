import unittest

import numpy as np
import vigra

from numpy.testing import assert_almost_equal

from lazyflow.operators import OpConnectedComponents
from lazyflow.operators.opBlockedConnectedComponents import\
    module_available as blocked_available
from lazyflow.graph import Graph


class TestOpConnectedComponents(unittest.TestCase):
    def setUp(self):
        OpConnectedComponents.useBlocking = False

    def testSimple(self):
        vol = np.zeros((100, 110, 120, 1, 1), dtype=np.uint8)
        vol[20:30, 25:35, 72:82, ...] = 1
        vol = vigra.taggedView(vol, axistags='xyzct')

        op = OpConnectedComponents(graph=Graph())
        op.Input.setValue(vol)

        output = op.Output[...].wait()

        # assert that all pixels in block are labeled
        assert np.all(output[20:30, 25:35, 72:82, ...] > 0)

        # assert that all pixels in block are labeled the same
        assert_almost_equal(output[20:30, 25:35, 72:82, ...].var(), 0)

        # assert that all other pixels are labeled 0
        assert_almost_equal(output[20:30, 25:35, 72:82, ...].sum(), output.sum())

    def testStrangeAxes(self):
        vol = np.zeros((100, 110, 120, 1, 1), dtype=np.uint8)
        vol[20:30, 25:35, 72:82] = 1
        vol = vigra.taggedView(vol, axistags='xyzct')

        vol = vol.withAxes(*'yzctx')

        op = OpConnectedComponents(graph=Graph())
        op.Input.setValue(vol)

        output = op.Output[...].wait()

        output = vigra.taggedView(output, axistags='yzctx').withAxes(*'xyzct')

        # assert that all pixels in block are labeled
        assert np.all(output[20:30, 25:35, 72:82, ...] > 0)

        # assert that all pixels in block are labeled the same
        assert_almost_equal(output[20:30, 25:35, 72:82, ...].var(), 0)

        # assert that all other pixels are labeled 0
        assert_almost_equal(output[20:30, 25:35, 72:82, ...].sum(), output.sum())

    def testStrangeDtype(self):
        vol = np.zeros((100, 110, 120, 1, 1), dtype=np.uint32)
        vol[20:30, 25:35, 72:82] = 1
        vol = vigra.taggedView(vol, axistags='xyzct')

        op = OpConnectedComponents(graph=Graph())
        op.Input.setValue(vol)

        # need the squeeze for np.all()
        output = op.Output[...].wait().squeeze()

        # assert that all pixels in block are labeled
        assert np.all(output[20:30, 25:35, 72:82] > 0)

        # assert that all pixels in block are labeled the same
        assert_almost_equal(output[20:30, 25:35, 72:82].var(), 0)

        # assert that all other pixels are labeled 0
        assert_almost_equal(output[20:30, 25:35, 72:82].sum(), output.sum())

    def testBackground(self):
        self.skipTest("Test not implemented yet.")


@unittest.skipIf(not blocked_available, "Module blockedarray is missing.")
class TestOpConnectedComponentsBlocked(TestOpConnectedComponents):
    def setUp(self):
        OpConnectedComponents.useBlocking = True


class TestMisc(unittest.TestCase):
    def setUp(self):
        pass

    def testFactorize(self):
        from lazyflow.operators.opConnectedComponents import _factorize
        f = [2, 7, 17]
        n = np.prod(f)
        g = _factorize(n)
        for i in f:
            assert i in g, "{} vs. {}".format(f,g)
    
    def testCombine(self):
        from lazyflow.operators.opConnectedComponents import _combine
        f = _combine([2, 7, 17])
        g = [2, 7, 17, 14, 34, 119, 238]
        for i in g:
            assert i in f, "{} vs. {}".format(f,g)
        
    
