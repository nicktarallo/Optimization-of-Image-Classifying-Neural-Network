#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --job-name=Inference
#SBATCH --mem=10G
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu

lscpu
module load anaconda3/2024.06 cuda/12.3.0
source activate pytorch_env
echo "no amp"
python testing_cifar10.py --do_pruning
echo "amp"
python testing_cifar10.py --use_amp --do_pruning
