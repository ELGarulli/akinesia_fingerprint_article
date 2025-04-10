#!/bin/bash
#SBATCH --job-name=merge_feat
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./outputs/neukin_merge.o%j
#SBATCH --error=./error/neukin_merge.e%j


eval "$(/opt/conda/bin/conda shell.bash hook)"
source /etc/profile.d/conda.sh
conda activate neural_feat
python neural_kin_merge.py