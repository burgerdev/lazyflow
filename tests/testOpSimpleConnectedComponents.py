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
import vigra

import unittest

from lazyflow.graph import Graph
from lazyflow.operators.opSimpleConnectedComponents import\
    OpSimpleConnectedComponents
from lazyflow.utility.testing import assertEquivalentLabeling


class TestOpSimpleConnectedComponents(unittest.TestCase):
    def setUp(self):
        g = Graph()
        op = OpSimpleConnectedComponents(graph=g)
        op.Background.setValue(0)
        self.op = op

    def testSimpleLabeling2d(self):
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

        expected = np.asarray(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 0],
             [0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 2, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint32)

        def tag(x):
            return vigra.taggedView(x, axistags='xy')

        vol = tag(vol)
        expected = tag(expected)

        op = self.op
        op.Input.setValue(vol)

        full = op.Output[...].wait()
        assertEquivalentLabeling(full, expected)

        part = op.Output[1:3, 1:3].wait()
        part_exp = tag(np.asarray([[0, 1], [2, 0]], dtype=np.uint32))
        assertEquivalentLabeling(part, part_exp)

    def testSimpleLabeling3d(self):
        vol = np.asarray([
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 0],
             [0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=np.uint8)

        expected = np.asarray([
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 0, 0, 0, 0],
             [0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 2, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=np.uint32)

        def tag(x):
            return vigra.taggedView(x, axistags='xyz')

        vol = tag(vol)
        expected = tag(expected)

        op = self.op
        op.Input.setValue(vol)

        full = tag(op.Output[...].wait())
        assertEquivalentLabeling(full, expected)

        part = tag(op.Output[:, 1:3, 1:3].wait())
        part_exp = tag(np.asarray([[[0, 1], [2, 0]]], dtype=np.uint32))
        assertEquivalentLabeling(part, part_exp)

    def testBackground(self):
        for dt in (np.uint8, np.uint32, np.float32):
            vol = np.asarray(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 1, 0, 0, 0, 0],
                 [0, 1, 0, 1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 1, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=dt)

            expected = (1 - vol).astype(np.uint32)

            def tag(x):
                return vigra.taggedView(x, axistags='xy')

            vol = tag(vol)
            expected = tag(expected)

            op = self.op
            op.Input.setValue(vol)
            op.Background.setValue(vol[2, 1])

            full = op.Output[...].wait()
            assertEquivalentLabeling(full, expected)
