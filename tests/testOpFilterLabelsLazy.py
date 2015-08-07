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

from lazyflow.graph import Graph
from lazyflow.operators.opFilterLabelsLazy import OpFilterLabelsLazy
from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.roi import roiFromShape
from lazyflow.rtype import SubRegion

from lazyflow.utility.testing import OpArrayPiperWithAccessCount
from lazyflow.utility.testing import OpBigArraySimulator
from lazyflow.utility.testing import Timeout

from testOpLazyConnectedComponents import DirtyAssert
from testOpLazyConnectedComponents import PropagateDirtyCalled

class TestOpFilterLabelsLazy(unittest.TestCase):
    def setUp(self):
        self.chunkShape = (5, 5, 5)
        vol = np.zeros((20, 20, 20), dtype=np.uint32)
        # 8 pix in 1 chunk
        vol[1:3, 1:3, 1:3] = 1
        # 8 pix in 8 chunks
        vol[4:6, 4:6, 9:11] = 2
        # 64 pix in 1 chunk
        vol[15:19, 15:19, 15:19] = 3
        # 64 pix in 4 chunk
        vol[13:17, 13:17, 5:9] = 4

        self.vol = self.five(vol)

        g = Graph()
        self.g = g
        op = OpArrayPiperWithAccessCount(graph=g)
        op.Input.setValue(self.vol)
        self.counter = op
        op = OpFilterLabelsLazy(graph=g)
        op.ChunkShape.setValue(self.chunkShape)
        op.Input.connect(self.counter.Output)
        self.op = op

    def five(self, arr, tags='xyz', axes='txyzc'):
        arr = vigra.taggedView(arr, axistags=tags)
        return arr.withAxes(*'txyzc')

    def testPlain(self):
        op = self.op
        op.MinLabelSize.setValue(0)
        out = self.five(op.Output[...].wait(),
                        tags=op.Output.meta.axistags)

        np.testing.assert_array_equal(out,
                                      self.vol)

    def testBinaryOut(self):
        op = self.op
        op.MinLabelSize.setValue(0)
        op.BinaryOut.setValue(True)
        out = self.five(op.Output[...].wait(),
                        tags=op.Output.meta.axistags)

        np.testing.assert_array_equal(out,
                                      (self.vol > 0).astype(np.uint32))

    def testMinLabelSize(self):
        op = self.op
        op.MinLabelSize.setValue(8)
        out = self.five(op.Output[...].wait(),
                        tags=op.Output.meta.axistags)
        np.testing.assert_array_equal(out,
                                      self.vol)

        op.MinLabelSize.setValue(7)
        op.MinLabelSize.setValue(8)
        out = self.five(op.Output[0, :5, :5, :, 0].wait(),
                        tags=op.Output.meta.axistags)
        np.testing.assert_array_equal(out,
                                      self.vol[:1, :5, :5, :, :1])

        op.MinLabelSize.setValue(9)
        out = self.five(op.Output[...].wait(),
                        tags=op.Output.meta.axistags)

        ref = self.vol.copy()
        ref[ref==1] = 0
        ref[ref==2] = 0
        np.testing.assert_array_equal(out, ref)

    def testMaxLabelSize(self):
        op = self.op
        op.MinLabelSize.setValue(0)
        op.MaxLabelSize.setValue(64)
        out = self.five(op.Output[...].wait(),
                        tags=op.Output.meta.axistags)

        np.testing.assert_array_equal(out,
                                      self.vol)

        op.MaxLabelSize.setValue(63)
        out = self.five(op.Output[...].wait(),
                        tags=op.Output.meta.axistags)

        ref = self.vol.copy()
        ref[ref==3] = 0
        ref[ref==4] = 0
        np.testing.assert_array_equal(out, ref)

    def testLazyness(self):
        op = self.op
        op.MinLabelSize.setValue(0)
        out = self.five(op.Output[0, 0:5, 0:5, 0:5, 0].wait(),
                        tags=op.Output.meta.axistags)

        # 4 chunks have to be visited
        # -> 4 for single chunk handling, 3*2 for hyperplanes
        #    1 for result
        np.testing.assert_equal(self.counter.accessCount, 11)

    def testMultiDim(self):
        vol = self.vol.withAxes(*'xyz')
        vol5d = np.zeros((2,) + vol.shape + (2,), dtype=vol.dtype)
        for t in range(vol5d.shape[0]):
            for c in range(vol5d.shape[-1]):
                vol5d[t, ..., c] = vol
        vol5d = self.five(vol5d, tags='txyzc')

        op = self.op
        op.Input.disconnect()
        op.Input.setValue(vol5d)

        op.MinLabelSize.setValue(8)
        out = self.five(op.Output[:, :5, :5, :5, :].wait(),
                        tags=op.Output.meta.axistags)
        ref = vol5d[:, :5, :5, :5, :]
        np.testing.assert_array_equal(out, ref)

        op.MinLabelSize.setValue(9)
        out = self.five(op.Output[:, :5, :5, :5, :].wait(),
                        tags=op.Output.meta.axistags)
        ref = self.five(np.zeros_like(out),
                        tags=op.Output.meta.axistags)
        np.testing.assert_array_equal(out, ref)

    def testReallyBigInput(self):
        g = Graph()
        pipe = OpBigArraySimulator(graph=g)
        pipe.Shape.setValue((1, 10000, 10000, 10000, 1))
        # 1TB memory should be sufficient to test

        pipe.Input.setValue(self.vol)

        op = OpFilterLabelsLazy(graph=Graph())
        op.Input.connect(pipe.Output)
        op.MinLabelSize.setValue(5)
        op.ChunkShape.setValue((10, 10, 10))
        req = op.Output[:, 3:17, 9:18, 0:3, :]

        timeout = Timeout(2, req.wait)
        timeout.start()

    def testDirtyPropagation(self):
        vol = np.zeros((2, 20, 20, 20, 1), dtype=np.uint32)
        vol[0, 0, 0, 0, 0] = 1
        vol = vigra.taggedView(vol, axistags='txyzc')
        op = self.op
        op.Input.disconnect()
        op.Input.setValue(vol)
        op.MinLabelSize.setValue(3)

        op.Output[...].wait()

        dirty = DirtyAssert(graph=self.g)
        dirty.Input.connect(op.Output)
        dirty.start = (0, 0, 0, 0, 0)
        dirty.stop = (1,) + vol.shape[1:4] + (1,)

        roi = SubRegion(op.Input,
                        start=(0, 0, 0, 0, 0),
                        stop=(1, 5, 6, 7, 1))
        with self.assertRaises(PropagateDirtyCalled):
            op.Input.setDirty(roi)

        vol[0, 0:2, 0:2, 0, 0] = 1
        out = op.Output[0, 0:2, 0:2, 0, 0].wait()
        assert np.all(out > 0)


        dirty.start = (0, 0, 0, 0, 0)
        dirty.stop = vol.shape
        with self.assertRaises(PropagateDirtyCalled):
            op.ChunkShape.setValue((3, 4, 5))
        with self.assertRaises(PropagateDirtyCalled):
            op.MinLabelSize.setValue(17)

    def testParallel(self):
        shape = (100, 100, 100)

        vol = np.ones(shape, dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        g = Graph()

        piper = OpArrayPiperWithAccessCount(graph=g)
        piper.Input.meta.ideal_blockshape = (25, 25, 25)
        piper.Input.setValue(vol)

        op = OpFilterLabelsLazy(graph=g)
        op.Input.connect(piper.Output)
        op.MinLabelSize.setValue(1)
        op.MaxLabelSize.setValue(100**3-1)

        reqs = [op.Output[..., 0],
                op.Output[..., 0],
                op.Output[..., 99],
                op.Output[..., 99],
                op.Output[0, ...],
                op.Output[0, ...],
                op.Output[99, ...],
                op.Output[99, ...],
                op.Output[:, 0, ...],
                op.Output[:, 0, ...],
                op.Output[:, 99, ...],
                op.Output[:, 99, ...]]

        [r.submit() for r in reqs]

        out = [r.wait() for r in reqs]

        for x in out:
            np.testing.assert_array_equal(x, 0)
