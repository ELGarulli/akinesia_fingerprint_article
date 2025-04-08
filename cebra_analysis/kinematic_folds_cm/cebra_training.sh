#!/bin/bash
#SBATCH --job-name=cebra_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --output=./outputs/cebra_train.o%j
#SBATCH --error=./error/cebra_train.e%j

source /etc/profile.d/conda.sh
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate cebra_cuda

python ~/CEBRA_analysis/CEBRA_train/higher_offset/kinematic_folds_cm/kinematic_run_offset.py


