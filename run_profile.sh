#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --job-name=Profiling
#SBATCH --mem=10G
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=courses-gpu
#SBATCH --cpus-per-task=28

lscpu
module load anaconda3/2024.06 cuda/12.3.0 Nsight/2024.7.1
source activate pytorch_env

echo "no workers"

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o no_workers_batch_128_amp python profiling_cifar10.py --use_amp --batch_size 128 --num_workers 0


echo "16 workers"
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o 16_workers_batch_128_amp python profiling_cifar10.py --use_amp --batch_size 128 --num_workers 16
