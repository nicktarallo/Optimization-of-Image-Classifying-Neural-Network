import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import argparse
import torch.nn.utils.prune as prune
from project_utils import prune_model

# Microsoft Copilot was used as a tool for answering questions related to PyTorch syntax for this code.

# Parse command line arguments and store them in variables:
parser = argparse.ArgumentParser(description="test settings")

parser.add_argument("--use_amp", action="store_true", help="use amp in inference")
parser.add_argument("--num_workers", type=int, help="num workers for data loader (default 0)")
parser.add_argument("--use_non_blocking", action="store_true", help="pin memory and use non blocking host-to-device transfers")
parser.add_argument("--track_memory", action="store_true", help="print peak GPU memory usage for each batch size")
parser.add_argument("--do_pruning", action="store_true", help="prune 20% of weights")
args = parser.parse_args()

use_amp = args.use_amp
num_workers = args.num_workers if args.num_workers else 0
use_non_blocking = args.use_non_blocking
track_memory = args.track_memory
do_pruning = args.do_pruning


# Define the device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Transformations to resize images from 32x32 to 224x224, convert to tensor, and normalize to match the ImageNet transformations used when training ResNet-18
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model and replace last layer
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)

# Load fine-tuned weights
model.load_state_dict(torch.load('fine_tuned_model.pth'))
# Use unstructred pruning if desired:
if do_pruning:
    prune_model(model)
model.to(device)
model.eval()  # Set the model to evaluation mode
print("Model reloaded from 'fine_tuned_model.pth'")

# Load CIFAR-10 dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print('Pytorch threads:', torch.get_num_threads())

# Loop through batch sizes, doubling each time:
inference_batch_size = 1
while inference_batch_size <= len(test_dataset):

    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=inference_batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_non_blocking)
    # Benchmark inference on the test set
    start_time = time.time()
    correct = 0
    total = 0
    # Reset peak GPU memory usage for this batch size
    if track_memory:
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # Go through each batch and move to the GPU
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device, non_blocking=use_non_blocking), labels.to(device, non_blocking=use_non_blocking)

            # Perform inference
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
            # Check how many results were correct
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end_time = time.time()
    accuracy = 100 * correct / total
    inference_time = end_time - start_time

    print(f"Inference batch size: {inference_batch_size}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total Inference Time: {inference_time:.2f} seconds")
    print(f"Time per Image: {inference_time / total:.4f} seconds")
    if track_memory:
        print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    inference_batch_size *= 2
