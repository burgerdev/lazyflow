#!/bin/bash

NP=`cat /proc/cpuinfo |grep processor|wc -l`
PY=`which python2 || which python`

# echo "executing MPI benchmark ($NP runners)"
NUM_PROC=$NP mpirun -np `expr $NP + 1` ${PY} opParallelImplementationBenchmark.py "$@"
# echo "executing non-MPI benchmarks"
NUM_PROC=$NP ${PY} opParallelImplementationBenchmark.py "$@"

