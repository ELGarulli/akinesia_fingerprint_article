#!/bin/bash
#SBATCH --job-name=parallel_nm
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-184
#SBATCH --cpus-per-task=1
#SBATCH --output=./outputs/parallel_nm.o%j
#SBATCH --error=./error/parallel_nm.e%j

eval "$(/opt/conda/bin/conda shell.bash hook)"
source /etc/profile.d/conda.sh
conda activate neural_feat
echo ${SLURM_ARRAY_TASK_ID}
python parallel_run_neural_feat.py