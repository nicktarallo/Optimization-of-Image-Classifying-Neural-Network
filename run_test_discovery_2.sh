#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --job-name=Inference
#SBATCH --mem=10G
#SBATCH --output=v100_my_job_%j.out
#SBATCH --error=v100_my_job_%j.err
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --partition=gpu

lscpu
module load anaconda3/2022.05 cuda/11.7
source activate pytorch_env_discovery
echo "no amp"
python testing_cifar10.py
echo "amp"
python testing_cifar10.py --use_amp
