
import unittest

import numpy as np
import vigra

from numpy.testing import assert_almost_equal

from lazyflow.operators import OpBlockedConnectedComponents
from lazyflow.graph import Graph


class TestOpBlockedConnectedComponents(unittest.TestCase):
    def setUp(self):
        pass

    def testSimple(self):
        vol = np.zeros((100, 110, 120), dtype=np.uint8)
        vol[20:30, 25:35, 72:82] = 1
        vol = vigra.taggedView(vol, axistags='xyz')

        op = OpBlockedConnectedComponents(graph=Graph())
        op.BlockShape.setValue(np.asarray((10, 10, 10)))
        op.Input.setValue(vol)

        output = op.Output[...].wait()

        # assert that all pixels in block are labeled
        assert np.all(output[20:30, 25:35, 72:82] > 0)

        # assert that all pixels in block are labeled the same
        assert_almost_equal(output[20:30, 25:35, 72:82].var(), 0)

        # assert that all other pixels are labeled 0
        assert_almost_equal(output[20:30, 25:35, 72:82].sum(), output.sum())

    def testMemUsage(self):
        self.skipTest("Not available")

    def testStrangeAxes(self):
        vol = np.zeros((100, 110), dtype=np.uint8)
        vol[20:30, 25:35] = 1
        vol = vigra.taggedView(vol, axistags='yt')

        op = OpBlockedConnectedComponents(graph=Graph())
        op.BlockShape.setValue(np.asarray((1, 10, 1)))
        op.Input.setValue(vol)

        output = op.Output[...].wait()

        # assert that all pixels in block are labeled
        assert np.all(output[20:30, 25:35] > 0)

    @unittest.expectedFailure  # not working yet
    def testStrangeDtype(self):
        vol = np.zeros((100, 110, 120), dtype=np.float)
        vol[20:30, 25:35, :] = 1
        vol = vigra.taggedView(vol, axistags='xyz')

        op = OpBlockedConnectedComponents(graph=Graph())
        op.BlockShape.setValue(np.asarray((10, 10, 10)))
        op.Input.setValue(vol)

        output = op.Output[...].wait()

        # assert that all pixels in block are labeled
        assert np.all(output[20:30, 25:35, :] > 0)

    @unittest.expectedFailure  # not working yet
    def testPrimeBlockShape(self):
        vol = np.zeros((100, 110, 12), dtype=np.uint8)
        vol[20:30, 25:35, :] = 1
        vol = vigra.taggedView(vol, axistags='xyz')

        op = OpBlockedConnectedComponents(graph=Graph())
        op.BlockShape.setValue(np.asarray((17, 13, 3)))
        op.Input.setValue(vol)

        output = op.Output[...].wait()

        # assert that all pixels in block are labeled
        assert np.all(output[20:30, 25:35, :] > 0)
