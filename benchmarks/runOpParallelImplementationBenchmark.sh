#!/bin/bash

NP=`cat /proc/cpuinfo |grep processor|wc -l`
PY=`which python`

if [[ $1 == "mpi" ]]
then
echo "executing MPI benchmark ($NP runners)"
NUM_PROC=$NP mpirun -np `expr $NP + 1` ${PY} opParallelImplementationBenchmark.py
else
echo "executing non-MPI benchmarks"
NUM_PROC=$NP ${PY} opParallelImplementationBenchmark.py
fi

