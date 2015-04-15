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
import os
import sys

import numpy as np
import vigra
import h5py

from lazyflow.graph import Graph
from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.stype import Opaque

from lazyflow.operators.vigraOperators import OpGaussianSmoothing
from lazyflow.operators.ioOperators import OpStreamingHdf5Reader
from lazyflow.operators import OpReorderAxes

from lazyflow.operators.opParallel import OpMapParallel
from lazyflow.operators.opParallelImplementations import RequestStrategy
from lazyflow.operators.opParallelImplementations import MultiprocessingStrategy
from lazyflow.operators.opParallelImplementations import MPIStrategy

from lazyflow.request import Request
Request.reset_thread_pool(num_workers=1)


class SmoothingWorkflow(Operator):
    Input = InputSlot()
    sigma = InputSlot(value=1.0)
    Output = OutputSlot(stype=Opaque)

    def __init__(self, *args, **kwargs):
        super(SmoothingWorkflow, self).__init__(*args, **kwargs)
        self.op = OpGaussianSmoothing(parent=self)
        self.op.Input.connect(self.Input)
        self.op.sigma.connect(self.sigma)

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.op.Output.meta)

    def execute(self, slot, subindex, roi, result):
        self.op.Output.get(roi).wait()
        return None

    def propagateDirty(self, slot, subindex, roi, result):
        self.Output.setDirty(roi)


num_cores = int(os.environ["NUM_PROC"])

for dim_x_str in sys.argv[1:]:
    dim_x = int(dim_x_str)
    shape = 1, dim_x, 1000, 120, 1
    chunkShape = 1, 500, 500, 20, 1

    graph = Graph()

    def run(op):
        with h5py.File('/tmp/test_{}.h5'.format(dim_x), 'r') as h5file:
            reader = OpStreamingHdf5Reader(graph=graph)
            reader.Hdf5File.setValue(h5file)
            reader.InternalPath.setValue('/data')

            order = OpReorderAxes(graph=graph)
            order.Input.connect(reader.OutputImage)
            order.AxisOrder.setValue('txyzc')

            op.Input.connect(order.Output)
            op.sigma.setValue(1.0)
            op.Output[...].wait()

    def benchmarkRegular():
        Request.reset_thread_pool()
        op = SmoothingWorkflow(graph=graph)
        run(op)

    def benchmarkRequest():
        Request.reset_thread_pool()
        op = OpMapParallel(SmoothingWorkflow, "Output",
                           RequestStrategy(chunkShape),
                           graph=graph)
        run(op)

    def benchmarkMultiprocessing():
        Request.reset_thread_pool(num_workers=1)
        op = OpMapParallel(SmoothingWorkflow, "Output",
                           MultiprocessingStrategy(num_cores, chunkShape),
                           graph=graph)
        run(op)

    def benchmarkMPI():
        # need HDF5 streaming reader!
        Request.reset_thread_pool(num_workers=1)
        op = OpMapParallel(SmoothingWorkflow, "Output",
                           MPIStrategy(chunkShape),
                           graph=graph)
        run(op)

    from mpi4py import MPI
    my_rank = MPI.COMM_WORLD.rank
    justmpi = MPI.COMM_WORLD.size > 1

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
        if my_rank == 0:
            print("MPI: {:.3f}s".format(res))
