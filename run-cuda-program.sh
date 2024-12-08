#!/bin/bash

## run-cuda-program.sh
##
## This script can be used to compile and run a CUDA program to the
## DISI HPC cluster using the SLURM job scheduler.
##
## Remember to:
## - change the basename of the source file using the environment variable `EXE`
##
## To submit a job:
##      sbatch ./run-cuda-program.sh
##
## To check the queue:
##      squeue
##
## To cancel a job:
##      scancel <jobid>
##
## Last modified on 2024-09-12 by Moreno Marzolla

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:05:00
#SBATCH --output last.out

## TODO: change accordingly
# export EXE="cuda"

echo "=== Start CUDA program ==="
# nvcc "cuda-vecadd0.cu" -o "cuda-vecadd0" && srun "./cuda-vecadd0"
nvcc cuda.cu -o cuda
srun ./cuda 1000000 3 100
echo "=== End of Job ==="
