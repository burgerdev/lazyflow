import numpy as np
import vigra

import unittest

from lazyflow.graph import Graph
from lazyflow.operators import OpLabelVolume

class TestVigra(unittest.TestCase):

    def setUp(self):
        self.method = np.asarray(['vigra'], dtype=np.object)

    def testSimpleUsage(self):
        vol = np.random.randint(255, size=(1000,100,10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        op = OpLabelVolume(graph=Graph())
        op.Method.setValue(self.method)
        op.Input.setValue(vol)

        out = op.Output[...].wait()

        assert np.all(vol.shape==out.shape)

    def testCorrectLabeling(self):
        vol = np.zeros((1000,100,10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        vol[20:40,10:30, 2:4] = 1

        op = OpLabelVolume(graph=Graph())
        op.Method.setValue(self.method)
        op.Input.setValue(vol)

        out = op.Output[...].wait()
        tags = op.Output.meta.getTaggedShape()
        out = vigra.taggedView(out, axistags="".join([s for s in tags]))

        assert np.all(vol.shape==out.shape)
        assert np.all(vol == out)


def assertEquivalentLabeling(x,y):
    assert np.all(x.shape==y.shape)

