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

import lazyflow
from lazyflow.graph import Graph
from lazyflow.operator import Operator, OutputSlot
from lazyflow.operators.opBlockedArrayCache import OpBlockedArrayCache
from lazyflow.operators.opArrayCache import OpArrayCache
from lazyflow.operators.cacheMemoryManager import CacheMemoryManager, ravel

from unittest import TestCase
from threading import Lock
import time


class OpZero(Operator):
    Output = OutputSlot()
    _lock = Lock()
    count = 0

    def __init__(self, shape, *args, **kwargs):
        super(OpZero, self).__init__(*args, **kwargs)

        self.Output.meta.shape = shape
        self.Output.meta.dtype = np.uint8
        self.Output.meta.axistags = vigra.defaultAxistags('xyz')
        # -> 1GiB data

    def setupOutputs(self):
        pass

    def execute(self, slot, subindex, roi, result):
        with self._lock:
            self.count += 1
        result[:] = 0


class TestCacheMemoryManager(TestCase):
    def setUp(self):
        CacheMemoryManager.setRefreshInterval(.1)
        mgr = CacheMemoryManager()
        self.mgr = mgr

    def tearDown(self):
        pass

    def testBlockedManagement(self):
        '''
        correct handling of BlockedArrayCache 
        '''
        # We choose a chunk size such that two chunks fit into available memory
        # (one in cache operator, one here), but four don't fit.
        # Additionally, the chunks should be large enough that unit test
        # overhead does not break our assumption. CARE: overhead is very large
        # (64MB on small test system)
        n_MB = 128
        chunkShape = (n_MB, 1024, 1024)
        lazyflow.AVAILABLE_RAM_MB = 3*np.prod(chunkShape)/1024.0**2

        g = Graph()

        op1 = OpZero((1024,)*3, graph=g)
        op2 = OpBlockedArrayCache(graph=g)
        op2.Input.connect(op1.Output)
        op2.innerBlockShape.setValue(chunkShape)
        op2.outerBlockShape.setValue(chunkShape)
        op2.fixAtCurrent.setValue(False)

        x = op2.Output[:n_MB, ...].wait()
        c = op1.count
        time.sleep(1)
        x = op2.Output[:n_MB, ...].wait()
        assert c == op1.count, "did not cache correctly"

        y = op2.Output[n_MB:2*n_MB, ...].wait()
        time.sleep(1)

        c = op1.count
        x = op2.Output[:8, ...].wait()
        assert c < op1.count, "did not clean up correctly"

    def testCleanup(self):
        '''
        use a cache that does not fit in memory, assert cleanup
        '''
        g = Graph()
        chunkShape = (1024, 1024)
        lazyflow.AVAILABLE_RAM_MB = .5 * np.prod(chunkShape)/1024.0**2

        op1 = OpZero(chunkShape, graph=g)
        op2 = OpArrayCache(graph=g)
        op2.Input.connect(op1.Output)
        op2.blockShape.setValue(chunkShape)
        op2.fixAtCurrent.setValue(False)

        x = op2.Output[:8, ...].wait()
        c = op1.count
        x = op2.Output[:8, ...].wait()
        assert c == op1.count, "did not cache correctly"

        time.sleep(1)

        c = op1.count
        x = op2.Output[:8, ...].wait()
        assert c < op1.count, "did not clean up correctly"


class TestRavel(TestCase):
    def setUp(self):
        pass

    def testSimple(self):
        '''
        test simple use of ravel() method to flatten a tree
        '''
        class C:
            n = 0

            def __init__(self, *children):
                self.children = list(children)
                C.n += 1
                self._n = C.n

            def getChildren(self):
                return self.children

            def lastAccessTime(self):
                return self._n

        ccc1 = C()
        ccc2 = C()
        cc1 = C(ccc1, ccc2)
        cc2 = C()
        c1 = C()
        c2 = C(cc1, cc2)
        c = C(c1, c2)
        #
        #   c1     ccc1
        #  /      /
        # c    cc1
        #  \  /   \
        #   c2     ccc2
        #     \
        #      cc2

        expected = [c1, ccc1, ccc2, cc1, cc2, c2, c]
        got = ravel(c)
        assert all(a is b for a, b in zip(expected, got))
