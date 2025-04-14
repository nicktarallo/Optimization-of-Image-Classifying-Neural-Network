#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --job-name=Inference
#SBATCH --mem=10G
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --partition=gpu

lscpu
module load anaconda3/2022.05 cuda/11.7
source activate pytorch_env
echo "no amp"
python testing_cifar10.py --track_memory
echo "amp"
python testing_cifar10.py --use_amp --track_memory
