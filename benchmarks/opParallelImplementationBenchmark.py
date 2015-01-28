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

from timeit import timeit

import numpy as np
import vigra

from lazyflow.graph import Graph

from lazyflow.operators.vigraOperators import OpGaussianSmoothing

from lazyflow.operators.opParallel import OpMapParallel
from lazyflow.operators.opParallelImplementations import RequestStrategy
from lazyflow.operators.opParallelImplementations import MultiprocessingStrategy
from lazyflow.operators.opParallelImplementations import MPIStrategy

from lazyflow.request import Request
Request.reset_thread_pool(num_workers=1)

shape = 1, 500, 1000, 120, 1
chunkShape = 1, 500, 500, 20, 1
n = 6
x = np.random.randint(0, 255, size=shape)
x = vigra.taggedView(x, axistags='txyzc')


def benchmarkRegular():
    Request.reset_thread_pool()
    g = Graph()
    op = OpGaussianSmoothing(graph=g)
    op.Input.setValue(x)
    op.sigma.setValue(1.0)
    op.Output[...].wait()

def benchmarkRequest():
    Request.reset_thread_pool()
    g = Graph()
    op = OpMapParallel(OpGaussianSmoothing, "Output",
                       RequestStrategy(chunkShape),
                       graph=g)
    op.Input.setValue(x)
    op.sigma.setValue(1.0)
    op.Output[...].wait()

def benchmarkMultiprocessing():
    Request.reset_thread_pool(num_workers=1)
    g = Graph()
    op = OpMapParallel(OpGaussianSmoothing, "Output",
                       MultiprocessingStrategy(n, chunkShape),
                       graph=g)
    op.Input.setValue(x)
    op.sigma.setValue(1.0)
    op.Output[...].wait()

def benchmarkMPI():
    # need HDF5 streaming reader!
    Request.reset_thread_pool(num_workers=1)
    g = Graph()
    op = OpMapParallel(OpGaussianSmoothing, "Output",
                       MPIStrategy(chunkShape),
                       graph=g)
    op.Input.setValue(x)
    op.sigma.setValue(1.0)
    op.Output[...].wait()


justmpi = True

if __name__ == "__main__":
    if not justmpi:
        res = timeit("benchmarkRegular()", number=1,
                     setup="from __main__ import benchmarkRegular")
        print("Single thread: {:.3f}s".format(res))
    
        res = timeit("benchmarkRequest()", number=1,
                     setup="from __main__ import benchmarkRequest")
        print("Requests: {:.3f}s".format(res))
    
        res = timeit("benchmarkMultiprocessing()", number=1,
                     setup="from __main__ import benchmarkMultiprocessing")
        print("Multiprocessing: {:.3f}s".format(res))
    else:
        res = timeit("benchmarkMPI()", number=1,
                     setup="from __main__ import benchmarkMPI")
        from mpi4py import MPI
        r = MPI.COMM_WORLD.rank
        if r == 0:
            print("MPI: {:.3f}s".format(res))

