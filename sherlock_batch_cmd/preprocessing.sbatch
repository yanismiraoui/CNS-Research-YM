#!/bin/bash
#SBATCH --job-name=preprocess_abide
#SBATCH --time=72:00:00
#SBATCH -c 4
#SBATCH -p normal
#SBATCH --mem=8GB
#SBATCH --mail-user=ymiraoui@stanford.edu
#SBATCH --mail-type=ALL
ml python/3.9
ml load py-pytorch/2.0.0_py39
cd $SCRATCH
source pre_venv/bin/activate
cd CNS-Research-YM/xGW-GAT_forked/
python3 main.py --dataset="PRIVATE" --model_name=gatv2 --sample_selection