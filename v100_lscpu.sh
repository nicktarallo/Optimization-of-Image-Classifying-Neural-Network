#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=Inference
#SBATCH --mem=10G
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --partition=courses-gpu

echo "workers v100 amp"
lscpu


