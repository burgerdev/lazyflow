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

import os

import numpy as np
import vigra
import h5py
import unittest

from numpy.testing import assert_array_equal, assert_equal

from lazyflow.utility.testing import assertEquivalentLabeling
from lazyflow.utility.testing import OpArrayPiperWithAccessCount
from lazyflow.utility.testing import OpBigArraySimulator
from lazyflow.utility.testing import Timeout
from lazyflow.operators.opLazyConnectedComponents\
    import OpLazyConnectedComponents as OpLazyCC

from lazyflow.graph import Graph
from lazyflow.operator import Operator
from lazyflow.slot import InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.roi import is_fully_contained

from lazyflow.operators import OpArrayPiper, OpCompressedCache


class TestOpLazyCC(unittest.TestCase):

    def setUp(self):
        pass

    def testCorrectLabeling(self):
        vol = np.zeros((1000, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        vol[20:40, 10:30, 2:4] = 1

        op = OpLazyCC(graph=Graph())
        op.Input.meta.ideal_blockshape = (100, 10, 10)
        op.Input.setValue(vol)

        out = op.Output[...].wait()
        out = vigra.taggedView(out, axistags=op.Output.meta.axistags)

        assertEquivalentLabeling(vol, out)

    def testSingletonZ(self):
        vol = np.zeros((82, 70, 1), dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        blocks = np.zeros(vol.shape, dtype=np.uint8)
        blocks[30:50, 40:60, :] = 1
        blocks[60:70, 30:40, :] = 3
        blocks = vigra.taggedView(blocks, axistags='xyz')

        vol[blocks > 0] = 255

        op = OpLazyCC(graph=Graph())
        op.Input.meta.ideal_blockshape = (30, 25, 1)
        op.Input.setValue(vol)

        out = op.Output[...].wait()
        out = vigra.taggedView(out, axistags=op.Output.meta.axistags)
        np.set_printoptions(threshold=np.nan, linewidth=200)
        print(out[..., 0])
        print(blocks[..., 0])
        assertEquivalentLabeling(blocks, out)

    def testLazyness(self):
        g = Graph()
        vol = np.asarray(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xy').withAxes(*'xyz')
        chunkShape = (3, 3, 1)

        opCount = OpArrayPiperWithAccessCount(graph=g)
        opCount.Input.meta.ideal_blockshape = chunkShape
        opCount.Input.setValue(vol)

        opCache = OpCompressedCache(graph=g)
        opCache.Input.connect(opCount.Output)
        opCache.BlockShape.setValue(chunkShape)

        op = OpLazyCC(graph=g)
        op.Input.meta.ideal_blockshape = chunkShape
        op.Input.connect(opCache.Output)

        out = op.Output[:3, :3, :].wait()
        n = 3
        print(opCache.CleanBlocks.value)
        assert opCount.accessCount <= n,\
            "Executed {} times (allowed: {})".format(opCount.accessCount,
                                                     n)

    def testContiguousLabels(self):
        g = Graph()
        vol = np.asarray(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 0],
             [0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xy').withAxes(*'xyz')
        chunkShape = (3, 3, 1)

        op = OpLazyCC(graph=g)
        op.Input.meta.ideal_blockshape = chunkShape
        op.Input.setValue(vol)

        out1 = op.Output[:3, :3].wait()
        out2 = op.Output[7:, 7:].wait()
        print(out1.max(), out2.max())
        assert max(out1.max(), out2.max()) == 2

    def testConsistency(self):
        vol = np.zeros((1000, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')
        vol[:200, ...] = 1
        vol[800:, ...] = 1

        op = OpLazyCC(graph=Graph())
        op.Input.meta.ideal_blockshape = (100, 10, 10)
        op.Input.setValue(vol)

        out1 = op.Output[:500, ...].wait()
        out2 = op.Output[500:, ...].wait()
        assert out1[0, 0, 0] != out2[499, 0, 0]

    def testCircular(self):
        g = Graph()

        op = OpLazyCC(graph=g)
        op.Input.meta.ideal_blockshape = (3, 3)

        vol = np.asarray(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 1, 1, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        vol1 = vigra.taggedView(vol, axistags='xy')
        vol2 = vigra.taggedView(vol, axistags='yx')
        vol3 = vigra.taggedView(np.flipud(vol), axistags='xy')
        vol4 = vigra.taggedView(np.flipud(vol), axistags='yx')

        for v in (vol1,):  # vol2, vol3, vol4):
            op.Input.setValue(v)
            for x in [0, 3, 6]:
                for y in [0, 3, 6]:
                    if x == 3 and y == 3:
                        continue
                    op.Input.setDirty(slice(None))
                    out = op.Output[x:x+3, y:y+3].wait()
                    print(x, y)
                    print(out.squeeze())
                    assert out.max() == 1

    def testParallelConsistency(self):
        vol = np.zeros((1000, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')
        vol[:200, ...] = 1
        vol[800:, ...] = 1

        op = OpLazyCC(graph=Graph())
        op.Input.meta.ideal_blockshape = (100, 10, 10)
        op.Input.setValue(vol)

        req1 = op.Output[:50, :10, :]
        req2 = op.Output[950:, 90:, :]
        req1.submit()
        req2.submit()

        out1 = req1.wait()
        out2 = req2.wait()

        assert np.all(out1 != out2)

    @unittest.expectedFailure
    def testSetDirty(self):
        g = Graph()
        vol = np.zeros((200, 100, 10, 2))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyzt')
        vol[:200, ...] = 1

        op = OpLazyCC(graph=g)
        op.Input.meta.ideal_blockshape = (100, 20, 5)
        op.ChunkShape.setValue((100, 20, 5))
        op.Input.setValue(vol)

        opCheck = DirtyAssert(graph=g)
        opCheck.Input.connect(op.Output)

        out = op.Output[:100, :20, :5, :].wait()

        roi = SubRegion(op.Input,
                        start=(0, 0, 0, 0),
                        stop=(15, 16, 17, 1))
        opCheck.start = (0, 0, 0, 0)
        opCheck.stop = (200, 100, 10, 1)
        with self.assertRaises(PropagateDirtyCalled):
            op.Input.setDirty(roi)

    def testDirtyPropagation(self):
        g = Graph()
        vol = np.asarray(
            [[0, 0, 0, 0],
             [0, 0, 1, 1],
             [0, 1, 0, 1],
             [0, 1, 0, 1]], dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xy').withAxes(*'xyz')

        chunkShape = (2, 2, 1)

        opCache = OpCompressedCache(graph=g)
        opCache.Input.setValue(vol)
        opCache.BlockShape.setValue(chunkShape)

        op = OpLazyCC(graph=g)
        op.Input.meta.ideal_blockshape = chunkShape
        op.Input.connect(opCache.Output)

        out1 = op.Output[:2, :2, :].wait()
        assert np.all(out1 == 0)

        opCache.Input[0:1, 0:1, 0:1] = np.asarray([[[1]]], dtype=np.uint8)

        out2 = op.Output[:1, :1, :1].wait()
        print(out2)
        assert np.all(out2 > 0)

    @unittest.skipIf('TRAVIS' in os.environ, "too costly")
    def testFromDataset(self):
        shape = (500, 500, 500)

        vol = np.zeros(shape, dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='zxy')

        centers = [(45, 15), (45, 350), (360, 50)]
        extent = (10, 10)
        shift = (1, 1)
        zrange = np.arange(0, 20)
        zsteps = np.arange(5, 455, 50)

        for x, y in centers:
            for z in zsteps:
                for t in zrange:
                    sx = x+t*shift[0]
                    sy = y+t*shift[1]
                    vol[zsteps + t, sx-extent[0]:sx+extent[0], sy-extent[0]:sy+extent[0]] = 255

        vol = vol.withAxes(*'xyz')

        # all at once
        op = OpLazyCC(graph=Graph())
        op.Input.meta.ideal_blockshape = (64, 64, 64)
        op.Input.setValue(vol)
        op.ChunkShape.setValue((64, 64, 64))
        out1 = op.Output[...].wait()
        out2 = vigra.analysis.labelVolumeWithBackground(vol)
        assertEquivalentLabeling(out1.view(np.ndarray), out2.view(np.ndarray))

    @unittest.skipIf('TRAVIS' in os.environ, "too costly")
    def testFromDataset2(self):
        shape = (500, 500, 500)

        vol = np.zeros(shape, dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='zxy')

        centers = [(45, 15), (45, 350), (360, 50)]
        extent = (10, 10)
        shift = (1, 1)
        zrange = np.arange(0, 20)
        zsteps = np.arange(5, 455, 50)

        for x, y in centers:
            for z in zsteps:
                for t in zrange:
                    sx = x+t*shift[0]
                    sy = y+t*shift[1]
                    vol[zsteps + t, sx-extent[0]:sx+extent[0], sy-extent[0]:sy+extent[0]] = 255

        vol = vol.withAxes(*'xyz')

        # step by step
        op = OpLazyCC(graph=Graph())
        op.Input.meta.ideal_blockshape = (64, 64, 64)
        op.Input.setValue(vol)
        out1 = np.zeros(op.Output.meta.shape,
                        dtype=op.Output.meta.dtype)
        for z in reversed(range(500)):
            out1[..., z:z+1] = op.Output[..., z:z+1].wait()
        vigra.writeHDF5(out1, '/tmp/data.h5', 'data')
        out2 = vigra.analysis.labelVolumeWithBackground(vol)
        assertEquivalentLabeling(out1.view(np.ndarray), out2.view(np.ndarray))

    def testParallel(self):
        shape = (100, 100, 100)

        vol = np.ones(shape, dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        g = Graph()

        piper = OpArrayPiperWithAccessCount(graph=g)
        piper.Input.meta.ideal_blockshape = (25, 25, 25)
        piper.Input.setValue(vol)

        op = OpLazyCC(graph=g)
        op.Input.connect(piper.Output)

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
        for i in range(len(out)-1):
            try:
                assert_array_equal(out[i].squeeze(), out[i+1].squeeze())
            except AssertionError:
                print(set(op.Output[...].wait().flat))
                raise
        print("TOTAL REQUESTS TO PIPER: {}".format(piper.accessCount))
        # we have 4*4*4=64 chunks
        # for each chunk
        #   1 request for labeling the chunk
        #   3 times 2 requests for merging in each direction
        #   (worst case, the border parts have fewer merges)
        assert piper.accessCount <= 64*7

    def testMultiDimSame(self):
        vol = np.zeros((2, 10, 10, 1, 3))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='txyzc')

        vol[:, 3:7, 3:7, :] = 1

        op = OpLazyCC(graph=Graph())
        op.Input.meta.ideal_blockshape = (1, 5, 5, 1, 1)
        op.Input.setValue(vol)

        out = op.Output[...].wait()
        out = vigra.taggedView(out, axistags=op.Output.meta.axistags)
        assert_array_equal(out[0, ...], out[1, ...])

    def testMultiDimDiff(self):
        vol = np.zeros((2, 10, 10, 1, 3))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='txyzc')

        vol[0, 3:7, 3:7, :] = 1
        vol[1, 7:, 7:, :] = 1

        op = OpLazyCC(graph=Graph())
        op.Input.meta.ideal_blockshape = (1, 5, 5, 1, 1)
        op.Input.setValue(vol)

        out = op.Output[...].wait()
        out = vigra.taggedView(out, axistags=op.Output.meta.axistags)
        assert np.all(out[1, :7, :7, ...] == 0)

    def testStrangeDim(self):
        vol = np.zeros((2, 10, 10, 1, 3))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='txyzc')

        vol[:, 3:7, 3:7, :] = 1

        strangeVol = vol.withAxes(*'ytxcz')

        op = OpLazyCC(graph=Graph())
        op.Input.meta.ideal_blockshape = (5, 1, 5, 1, 1)
        op.Input.setValue(strangeVol)

        out = op.Output[...].wait()
        out = vigra.taggedView(out, axistags=op.Output.meta.axistags)
        out = out.withAxes(*'txyzc')
        assert np.all(out[1, 3:7, 3:7, ...] > 0)

    def testReallyBigInput(self):
        shape = (1, 10000, 10000, 10000, 1)
        # assumptions
        #   * 1TB does not fit in memory
        #   * several chunks of 1MB + management structures for 10M
        #     chunks fit in memory

        g = Graph()
        pipe = OpBigArraySimulator(graph=g)
        pipe.Shape.setValue(shape)

        im = np.asarray(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 1, 1, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

        im = im[..., np.newaxis]
        im2 = im*0
        cc = (im2, im, im, im2)
        vol = np.concatenate(cc, axis=2)
        assert_array_equal(vol.shape, (9, 9, 4))
        vol = vigra.taggedView(vol, axistags='xyz')
        vol = vol.withAxes(*'txyzc')

        pipe.Input.setValue(vol)

        # test if value provider works correctly
        out = pipe.Output[:, 3:17, 9:18, 0:4, :].wait()
        out_cut = out[:, 0:6, ...].squeeze()
        expected = vol.view(np.ndarray)[:, 3:9, ...].squeeze()
        assert_array_equal(out_cut, expected)

        class MyOpArrayPiper(OpArrayPiperWithAccessCount):
            def execute(self, slot, subindex, roi, result):
                super(MyOpArrayPiper, self).execute(
                    slot, subindex, roi, result)
                with self._lock:
                    if not hasattr(self, "rois"):
                        self.rois = []
                    self.rois.append(roi)

        count = MyOpArrayPiper(graph=g)
        count.Input.connect(pipe.Output)

        op = OpLazyCC(graph=g)
        op.Input.connect(count.Output)
        req = op.Output[:, 3:17, 9:18, 0:3, :]

        # test took approximately one second on an Intel Atom ...
        timeout = Timeout(2, req.wait)
        timeout.start()
        for roi in count.rois:
            assert is_fully_contained((roi.start, roi.stop),
                                      ((0,)*5, (1, 27, 27, 8, 1))),\
                str(roi)


class DirtyAssert(Operator):
    Input = InputSlot()

    def propagateDirty(self, slot, subindex, roi):
        assert_array_equal(roi.start, self.start)
        assert_array_equal(roi.stop, self.stop)
        raise PropagateDirtyCalled()


class PropagateDirtyCalled(Exception):
    pass
