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


class OpExecuteCounter(Operator):
    Input = InputSlot()
    Output = OutputSlot()

    def setupOutputs(self):
        self.count = 0
        self.Output.meta.assignFrom(self.Input.meta)

    def execute(self, slot, subindex, roi, result):
        self.count += 1
        req = self.Input.get(roi)
        req.writeInto(result)
        req.block()

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(roi)


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
        op = OpExecuteCounter(graph=g)
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
        np.testing.assert_equal(self.counter.count, 11)

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
