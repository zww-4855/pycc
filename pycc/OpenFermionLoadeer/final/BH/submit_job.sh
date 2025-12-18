#!/bin/bash
#SBATCH -A CSC635
#SBATCH -J BH 
#SBATCH -o %x-%j.out
#SBATCH -t 24:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mem=10GB

module load openblas/0.3.23-omp
module load netlib-lapack
module load cmake
module load python

source activate /ccsopen/home/zwu
export PYTHONPATH=$PYTHONPATH:/gpfs/wolf2/cades/csc635/proj-shared/.xacc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

conda activate pyCCdev

python3 -u new.py > out


source deactivate
