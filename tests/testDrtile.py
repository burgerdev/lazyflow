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

import time

import numpy as np
import vigra
import unittest

from lazyflow.drtile2 import drtile


class TestDrtile(unittest.TestCase):
    def setUp(self):
        pass

    def test1d(self):
        x = [1, 1, 0, 1, 0, 0, 1, 1, 1]
        x = np.asarray(x)

        rois = drtile(x)
        np.testing.assert_equal(len(rois), 3)

    def testSimple2d(self):
        x = [[1, 1, 1, 1, 1],
             [1, 0, 0, 0, 1],
             [1, 0, 0, 0, 1],
             [1, 1, 1, 1, 1]]
        x = np.asarray(x)

        rois = drtile(x)
        np.testing.assert_equal(len(rois), 4)

    @unittest.expectedFailure
    def testHard2d(self):
        x = [[1, 1, 1, 1, 1],
             [1, 1, 0, 0, 1],
             [1, 1, 0, 0, 1],
             [0, 1, 1, 1, 1]]
        x = np.asarray(x)

        rois = drtile(x)
        np.testing.assert_equal(len(rois), 4)

    def test3d(self):
        x = [[1, 1, 1, 1, 1],
             [1, 0, 0, 0, 1],
             [1, 0, 0, 0, 1],
             [1, 1, 1, 1, 1]]
        x = np.asarray(x)

        
        y = [[1, 0, 0, 0, 1],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [1, 0, 0, 0, 1]]
        y = np.asarray(y)

        vol = np.concatenate((x[:, :, np.newaxis],
                              y[:, :, np.newaxis],
                              y[:, :, np.newaxis],
                              x[:, :, np.newaxis]), axis=2)

        rois = drtile(vol)

        print(vol.shape)
        print("ARRAY")
        print(vol)
        print("TILING")
        print(tilingToArray(rois, vol.shape))

        # 2*4 in x arrays and 4 in 2*y
        np.testing.assert_equal(len(rois), 3*4)


def tilingToArray(x, shape):
    x = np.asarray(x)
    y = np.zeros(shape, dtype=np.uint32)
    half = x.shape[1]//2
    for i in range(x.shape[0]):
        start = x[i, :half]
        stop = x[i, half:]
        key = tuple([slice(a, b) for a, b in zip(start, stop)])
        y[key] = i+1
    return y


class BenchmarkTimer(object):
    def __init__(self, benchmarkName, n):
        self._n = n
        self._s = benchmarkName
    def __enter__(self):
        self._t = time.time()

    def __exit__(self, *args, **kwargs):
        t = time.time()
        e = (t-self._t)
        print("Elapsed time ({}): total={}s, per-pixel={}s"
              "".format(self._s, self._sci(e), self._sci(e/self._n)))

    def _sci(self, x):
        e = 0
        rev = False
        if x > 1:
            x = 1/float(x)
            rev = True
        while x < 1:
            e += 3
            x *= 1e3
        if rev:
            x = 1/x
            x *= 1e3
            e -= 3
        else:
            e = -e
        return "{:.2f}".format(x) + ("*10^{}".format(e) if e else "")

if __name__ == "__main__":
    from lazyflow.drtile2.drtile import _drtile_old, _drtile_new

    for n in (5, 10, 20, 30, 40, 50, 100, 1000):
        shape = (n,)*2
        print("Shape {}".format(shape))
        np.random.seed(0)
        x = np.random.randint(0, 2, size=shape)

        for foo, name in [(_drtile_old, "drtile.cpp"),
                          (_drtile_new, "drtile.py")]:
            with BenchmarkTimer(name, x.size):
                roi = foo(x)
