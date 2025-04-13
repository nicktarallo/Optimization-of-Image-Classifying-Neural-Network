import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import argparse

parser = argparse.ArgumentParser(description="test settings")

parser.add_argument("--batch_size", type=int, help="batch size for inference")
parser.add_argument("--use_amp", action="store_true", help="use amp in inference")
parser.add_argument("--num_workers", type=int, help="num workers for data loader (default 0)")
parser.add_argument("--use_non_blocking", action="store_true", help="pin memory and use non blocking host-to-device transfers")
args = parser.parse_args()

inference_batch_size = args.batch_size if args.batch_size else 32
use_amp = args.use_amp
num_workers = args.num_workers if args.num_workers else 0
use_non_blocking = args.use_non_blocking

# Define the device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define transformations (normalize CIFAR-10 images and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 to fit ResNet-18
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Reload the model ---
# Recreate the model architecture (same as the one we trained)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)

# Load the saved model weights
model.load_state_dict(torch.load('fine_tuned_model.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode
print("Model reloaded from 'fine_tuned_model.pth'")

# Load CIFAR-10 dataset
# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print('Pytorch threads:', torch.get_num_threads())

inference_batch_size = 1
while inference_batch_size <= len(test_dataset):

    # Create DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=inference_batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_non_blocking)
    # Benchmark inference on the test set
    start_time = time.time()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device, non_blocking=use_non_blocking), labels.to(device, non_blocking=use_non_blocking)

            # Perform inference
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
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
    inference_batch_size *= 2
