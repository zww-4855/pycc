#!/bin/bash
#SBATCH --job-name=a2_tz
#SBATCH --output=zzz.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=9gb
#SBATCH --qos=bartlett-b

echo $SLURM_NODELIST
echo $SLURM_JOBID

module unload intel
module unload openmpi
module load gcc/12.2.0
module load mkl/2020.0.166

export PATH=/blue/bartlett/z.windom/TBLIS:$PATH

xcfour > out.out

