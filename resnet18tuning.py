import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import torch.optim as optim

# Resize, convert to tensor, and normalize to match ImageNet transformations used for training ResNet-18
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Get CIFAR-10 Dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use batch size 32 for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet-18
model = models.resnet18(pretrained=True)

# Freeze layers that are not the fully-connected layer
for param in model.parameters():
    param.requires_grad = False

# Modify the last layer to have 10 output classes instead of 1,000
model.fc = nn.Linear(model.fc.in_features, 10)

# Move model to GPU (or CPU)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)  # Optimizer only uses fully connected layer

# Fine-tune the model on CIFAR-10
num_epochs = 10
for epoch in range(num_epochs):
    print(f'epoch {epoch}')
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # Go through batches
    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        # Compute loss
        loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print statistics for this epoch
    epoch_accuracy = 100 * correct / total
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_model.pth')
print("Model saved to 'fine_tuned_model.pth'")

# Evaluate on the test set
model.eval()  # Set the model to evaluation mode
start_time = time.time()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Perform inference
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

end_time = time.time()
accuracy = 100 * correct / total
inference_time = end_time - start_time

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Total Inference Time: {inference_time:.2f} seconds")
print(f"Time per Image: {inference_time / total:.4f} seconds")

