#!/bin/bash
#SBATCH --job-name=parcel_difumo_abide
#SBATCH --time=48:00:00
#SBATCH -c 4
#SBATCH -p normal
#SBATCH --mem=32GB
#SBATCH --mail-user=ymiraoui@stanford.edu
#SBATCH --mail-type=ALL
ml python/3.9
ml load py-pytorch/2.0.0_py39
cd $SCRATCH
cd CNS-Research-YM/
python3 parcel_difumo_abide.py ./abide/func_preproc/ 1