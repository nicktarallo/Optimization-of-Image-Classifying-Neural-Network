#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --job-name=FineTuning
#SBATCH --mem=10G
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu

module load anaconda3/2024.06 cuda/12.3.0
source activate pytorch_env
python resnet18tuning.py

