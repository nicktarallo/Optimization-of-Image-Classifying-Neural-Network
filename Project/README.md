# Optimizing Inference of Image Classifying Neural Network

### All code should be run on Explorer

Important packages include `torch` and `torchvision`. These will be installed when creating the conda environment as described later.

### Running inference:

First, navigate to the `Project` folder if you are not in it already

```bash
cd Project
```

To run the code, you must create a Conda environment on Explorer.

To do this, get a P100 node and set up the environment as follows. These are adapted from the instructions in the Research Computing documentation (https://rc-docs.northeastern.edu/en/latest/gpus/gpujobsubmission.html):

```bash
srun --partition=courses-gpu --nodes=1 --gres=gpu:p100:1 --cpus-per-task=2 --mem=10GB --time=02:00:00 --pty /bin/bash
module load anaconda3/2024.06 cuda/12.3.0
conda create --name pytorch_env -c conda-forge python=3.10 -y
source activate pytorch_env
conda install jupyterlab -y
pip3 install torch torchvision torchaudio
```

From there, the code can be run. The main code to run inference tests is `testing_cifar10.py`. This code will run through batch sizes from 1 to 8192 and perform inference on the entire CIFAR-10 test set

There are multiple optional command line arguments for this file:
- `--use_amp`: This flag will cause Automatic Mixed Precision to be utilized
- `--num_workers [number]`: This will set the number of workers to be used for multi-process data loading
- `--use_non_blocking`: This will make the DataLoader use pinned memory and use non blocking transfers to the GPU
- `--track_memory`: This will make the peak GPU memory utilization show for each batch
- `--do_pruning`: This will prune the model with unstructured pruning (not recommended, does not help)

To run the program, acquire a GPU node with either a P100 or V100. It is recommended to use `--cpus-per-task=28` when acquiring the node with srun or sbatch. On the GPU node, run:

```bash
module load anaconda3/2024.06 cuda/12.3.0
source activate pytorch_env
python testing_cifar10.py [INCLUDE ANY FLAGS/COMMAND LINE ARGUMENTS HERE]
``` 

For example to run with AMP and 4 workers, we could use:

```bash
python testing_cifar10.py --use_amp --num_workers 4
```

This code assumes that `fine_tuned_model.pth` is in the same folder that you are executing from. This should already be the case assuming that you are running directly from the `Project` folder.

When running the code, it will automatically be saved in a `./data` folder that will be created from `torchvision` automatically downloading the CIFAR-10 dataset.

### Running inference with profiling:

To run code that will generate a file that can be viewed on Nsight systems, run the `profiling_cifar10.py` file instead.

This file has all of the same command line arguments as above as well as another one:
- `--batch_size [NUMBER]`: Set the batch size to the specified number. Defaults to 128

In comparison to `testing_cifar10.py`, this file only runs for a single batch size. It will produce an `nsys-rep` file.

To run this file, once again, a node with a P100 or V100 must be acquired. Then, run the following commands:

```bash
module load anaconda3/2024.06 cuda/12.3.0 Nsight/2024.7.1

source activate pytorch_env

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o no_workers_batch_128_amp python profiling_cifar10.py --use_amp --batch_size 128 --num_workers 0
```

The flags for the Python program can be modified at the end of the `nsys profile` command.
The name of the resulting `nsys-rep` file can be modified by changing what is after the -o in the command.

The command is sourced from: https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59

### Training:

There is already a fully pre-trained and fine-tuned model in the file `fine_tuned_model.pth`. If you want to rerun training, you can acquire a GPU node and run the commands:

```bash
module load anaconda3/2024.06 cuda/12.3.0 Nsight/2024.7.1
source activate pytorch_env
python resnet18tuning.py
```

This will run for 10 epochs and save the resulting model in `fine_tuned_model.pth`.

### Batch files:

There are some example sbatch files that can be used to run various experiments that were performed in the writeup. These are located in the /scripts folder. Unmarked ones use the P100 GPU, while file names marked with `_v100` use the V100 GPU. An example of how to use one is here:

```bash
sbatch scripts/run_test.sh
```

The `useful_outputs` folder contains a lot of outputs that were used when collecting data for experiments.

Microsoft Copilot was used as a tool for answering questions related to PyTorch syntax for this code.
