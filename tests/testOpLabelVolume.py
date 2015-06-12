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
import vigra

import unittest
import weakref
import gc

from lazyflow.graph import Graph
from lazyflow.operators import OpLabelVolume, OpArrayPiper
from lazyflow.operator import Operator
from lazyflow.slot import InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.utility.testing import assertEquivalentLabeling
from lazyflow.operators.cacheMemoryManager import CacheMemoryManager

from numpy.testing import assert_array_equal


class TestVigra(unittest.TestCase):

    def setUp(self):
        self.method = np.asarray(['vigra'], dtype=np.object)

    def testSimpleUsage(self):
        vol = np.random.randint(255, size=(100, 30, 4))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        op = OpLabelVolume(graph=Graph())
        op.Method.setValue(self.method)
        op.Input.setValue(vol)

        out = op.Output[...].wait()

        assert_array_equal(vol.shape, out.shape)

    def testCorrectLabeling(self):
        vol = np.zeros((200, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        vol[20:40, 10:30, 2:4] = 1

        op = OpLabelVolume(graph=Graph())
        op.Method.setValue(self.method)
        op.Input.setValue(vol)

        out = op.Output[...].wait()
        tags = op.Output.meta.getTaggedShape()
        out = vigra.taggedView(out, axistags="".join([s for s in tags]))

        assertEquivalentLabeling(vol, out)

    def testMultiDim(self):
        vol = np.zeros((82, 70, 75, 5, 5), dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xyzct')

        blocks = np.zeros(vol.shape, dtype=np.uint8)
        blocks[30:50, 40:60, 50:70, 2:4, 3:5] = 1
        blocks[30:50, 40:60, 50:70, 2:4, 0:2] = 2
        blocks[60:70, 30:40, 10:33, :, :] = 3

        vol[blocks == 1] = 255
        vol[blocks == 2] = 255
        vol[blocks == 3] = 255

        op = OpLabelVolume(graph=Graph())
        op.Method.setValue(self.method)
        op.Input.setValue(vol)

        out = op.CachedOutput[...].wait()
        tags = op.CachedOutput.meta.getTaggedShape()
        print(tags)
        out = vigra.taggedView(out, axistags="".join([s for s in tags]))

        for c in range(out.shape[3]):
            for t in range(out.shape[4]):
                print("t={}, c={}".format(t, c))
                assertEquivalentLabeling(blocks[..., c, t], out[..., c, t])

    def testSingletonZ(self):
        vol = np.zeros((82, 70, 1, 5, 5), dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xyzct')

        blocks = np.zeros(vol.shape, dtype=np.uint8)
        blocks[30:50, 40:60, :, 2:4, 3:5] = 1
        blocks[30:50, 40:60, :, 2:4, 0:2] = 2
        blocks[60:70, 30:40, :, :, :] = 3

        vol[blocks == 1] = 255
        vol[blocks == 2] = 255
        vol[blocks == 3] = 255

        op = OpLabelVolume(graph=Graph())
        op.Method.setValue(self.method)
        op.Input.setValue(vol)

        out = op.CachedOutput[...].wait()
        tags = op.CachedOutput.meta.getTaggedShape()
        print(tags)
        out = vigra.taggedView(out, axistags="".join([s for s in tags]))

        for c in range(out.shape[3]):
            for t in range(out.shape[4]):
                print("t={}, c={}".format(t, c))
                assertEquivalentLabeling(blocks[..., c, t], out[..., c, t])

    def testConsistency(self):
        vol = np.zeros((100, 50, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')
        vol[:20, ...] = 1
        vol[80:, ...] = 1

        op = OpLabelVolume(graph=Graph())
        op.Method.setValue(self.method)
        op.Input.setValue(vol)

        out1 = op.CachedOutput[:50, ...].wait()
        out2 = op.CachedOutput[50:, ...].wait()
        assert out1[0, 0, 0] != out2[49, 0, 0]

    def testNoRecomputation(self):
        g = Graph()

        vol = np.zeros((100, 50, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')
        vol[:20, ...] = 1
        vol[80:, ...] = 1

        opCount = CountExecutes(graph=g)
        opCount.Input.setValue(vol)

        op = OpLabelVolume(graph=g)
        op.Method.setValue(self.method)
        op.Input.connect(opCount.Output)

        op.CachedOutput[:50, ...].wait()
        op.CachedOutput[50:, ...].wait()

        assert opCount.numExecutes == 1

    def testCorrectBlocking(self):
        g = Graph()
        c, t = 2, 3
        vol = np.zeros((100, 50, 10, 2, 3))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyzct')
        vol[:20, ...] = 1
        vol[80:, ...] = 1

        opCount = CountExecutes(graph=g)
        opCount.Input.setValue(vol)

        op = OpLabelVolume(graph=g)
        op.Method.setValue(self.method)
        op.Input.connect(opCount.Output)

        op.CachedOutput[:50, ...].wait()
        op.CachedOutput[50:, ...].wait()

        assert opCount.numExecutes == c*t

    def testThreadSafety(self):
        g = Graph()

        vol = np.zeros((1000, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')
        vol[:200, ...] = 1
        vol[800:, ...] = 1

        opCount = CountExecutes(graph=g)
        opCount.Input.setValue(vol)

        op = OpLabelVolume(graph=g)
        op.Method.setValue(self.method)
        op.Input.connect(opCount.Output)

        reqs = [op.CachedOutput[...] for i in range(4)]
        [r.submit() for r in reqs]
        [r.block() for r in reqs]
        assert opCount.numExecutes == 1,\
            "Parallel requests to CachedOutput resulted in recomputation "\
            "({}/4)".format(opCount.numExecutes)

        # reset numCounts
        opCount.numExecutes = 0

        reqs = [op.Output[250*i:250*(i+1), ...] for i in range(4)]
        [r.submit() for r in reqs]
        [r.block() for r in reqs]
        assert opCount.numExecutes == 4,\
            "Not all requests to Output were computed on demand "\
            "({}/4)".format(opCount.numExecutes)

    def testSetDirty(self):
        g = Graph()
        vol = np.zeros((5, 2, 200, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='tcxyz')

        op = OpLabelVolume(graph=g)
        op.Method.setValue(self.method)
        op.Input.setValue(vol)

        opCheck = DirtyAssert(graph=g)
        opCheck.Input.connect(op.Output)
        opCheck.willBeDirty(1, 1)

        roi = SubRegion(op.Input,
                        start=(1, 1, 0, 0, 0),
                        stop=(2, 2, 200, 100, 10))
        with self.assertRaises(PropagateDirtyCalled):
            op.Input.setDirty(roi)

        opCheck.Input.disconnect()
        opCheck.Input.connect(op.CachedOutput)
        opCheck.willBeDirty(1, 1)

        op.Output[...].wait()

        roi = SubRegion(op.Input,
                        start=(1, 1, 0, 0, 0),
                        stop=(2, 2, 200, 100, 10))
        with self.assertRaises(PropagateDirtyCalled):
            op.Input.setDirty(roi)

    def testUnsupported(self):
        g = Graph()
        vol = np.zeros((50, 50))
        vol = vol.astype(np.int16)
        vol = vigra.taggedView(vol, axistags='xy')
        vol[:200, ...] = 1
        vol[800:, ...] = 1

        op = OpLabelVolume(graph=g)
        op.Method.setValue(self.method)
        with self.assertRaises(ValueError):
            op.Input.setValue(vol)

    def testBackground(self):
        vol = np.zeros((200, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        vol[20:40, 10:30, 2:4] = 1

        op = OpLabelVolume(graph=Graph())
        op.Method.setValue(self.method)
        op.Background.setValue(1)
        op.Input.setValue(vol)

        out = op.Output[...].wait()
        tags = op.Output.meta.axistags
        out = vigra.taggedView(out, axistags=tags)

        assert np.all(out[20:40, 10:30, 2:4] == 0)
        assertEquivalentLabeling(1-vol, out)

        vol = vol.withAxes(*'xyzct')
        vol = np.concatenate(3*(vol,), axis=3)
        vol = np.concatenate(4*(vol,), axis=4)
        vol = vigra.taggedView(vol, axistags='xyzct')
        assert len(vol.shape) == 5
        assert vol.shape[3] == 3
        assert vol.shape[4] == 4


class TestLazy(TestVigra):

    def setUp(self):
        self.method = np.asarray(['lazy'], dtype=np.object)

    @unittest.skip("This test does not make sense with lazy connected components")
    def testCorrectBlocking(self):
        pass

    @unittest.skip("This test does not make sense with lazy connected components")
    def testNoRecomputation(self):
        pass

    def testThreadSafety(self):
        g = Graph()

        vol = np.zeros((1000, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')
        vol[:200, ...] = 1
        vol[800:, ...] = 1

        opCount = CountExecutes(graph=g)
        opCount.Input.setValue(vol)
        opCount.Output.meta["ideal_blockshape"] = vol.shape

        op = OpLabelVolume(graph=g)
        op.Method.setValue(self.method)
        op.Input.connect(opCount.Output)

        reqs = [op.CachedOutput[...] for i in range(4)]
        [r.submit() for r in reqs]
        [r.block() for r in reqs]
        assert opCount.numExecutes == 1,\
            "Parallel requests to CachedOutput resulted in recomputation "\
            "({}/4)".format(opCount.numExecutes)


class DirtyAssert(Operator):
    Input = InputSlot()

    def willBeDirty(self, t, c):
        self._t = t
        self._c = c

    def propagateDirty(self, slot, subindex, roi):
        t_ind = self.Input.meta.axistags.index('t')
        c_ind = self.Input.meta.axistags.index('c')
        assert roi.start[t_ind] == self._t
        assert roi.start[c_ind] == self._c
        assert roi.stop[t_ind] == self._t+1
        assert roi.stop[c_ind] == self._c+1
        raise PropagateDirtyCalled()


class CountExecutes(Operator):
    Input = InputSlot()
    Output = OutputSlot()

    numExecutes = 0

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi)

    def execute(self, slot, sunbindex, roi, result):
        self.numExecutes += 1
        req = self.Input.get(roi)
        req.writeInto(result)
        req.block()


class PropagateDirtyCalled(Exception):
    pass


if __name__ == "__main__":
    method = np.asarray(['lazy'], dtype=np.object)
    vol = np.random.randint(255, size=(10, 10, 10))
    vol = vol.astype(np.uint8)
    vol = vigra.taggedView(vol, axistags='xyz')

    op = OpLabelVolume(graph=Graph())
    op.Method.setValue(method)
    op.Input.setValue(vol)

    out = op.Output[...].wait()

    assert_array_equal(vol.shape, out.shape)
