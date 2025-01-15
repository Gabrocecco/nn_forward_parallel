#!/bin/bash

## run-omp-program.sh
##
## This script can be used to submit an OpenMP program to the DISI HPC
## cluster using the SLURM job scheduler.
##
## Remember to:
## - redefine `--cpus-per-task` with the number of cores you want to use
## - change the name of the executable using the environment variable `EXE`
##
## To submit a job:
##      sbatch ./run-omp-program.sh
##
## To submit a job specifying the number of threads
## (bypassws the default value specified by --cpus-per-task)
##
##      sbatch -c 2 ./run-omp-program.sh
##
## To check the queue:
##      squeue
##
## To cancel a job:
##      scancel <jobid>
##
## Last modified in 2024-09-12 by Moreno Marzolla

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 4
#SBATCH --time=0-00:05:00
#SBATCH --output last.out

## TODO: change accordingly

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== Start $EXE with $OMP_NUM_THREADS threads ==="
srun gcc -fopenmp omp.c -o omp -lm -std=c99 -Wall -Wpedantic
srun ./omp 10000000 3 100
echo "=== End of Job ==="
