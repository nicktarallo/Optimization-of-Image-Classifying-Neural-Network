#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:40:00
#SBATCH --job-name=Inference
#SBATCH --mem=10G
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=28

echo "workers v100 amp"
lscpu
module load anaconda3/2024.06 cuda/12.3.0
source activate pytorch_env
numbers=(0 1 2 4 8 16 28 56)

# Loop through each number and use it as an argument
for number in "${numbers[@]}"; do
  echo "$number workers"
  python testing_cifar10.py --use_amp --num_workers "$number"
done

