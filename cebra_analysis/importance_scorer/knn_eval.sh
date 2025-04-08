#!/bin/bash
#SBATCH --job-name=parallel_knn
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-11
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=./outputs/parallel_knn.o%j
#SBATCH --error=./error/parallel_knn.e%j


source /etc/profile.d/conda.sh
#eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate cebra_cuda
echo ${SLURM_ARRAY_TASK_ID}
python ~/CEBRA_analysis/CEBRA_train/higher_offset/importance_scorer/knn_eval_kin.py
python ~/CEBRA_analysis/CEBRA_train/higher_offset/importance_scorer/knn_eval.py
