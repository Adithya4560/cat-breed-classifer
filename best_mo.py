import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np

# Set seed
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Transforms (for EfficientNet-B0)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet-B0 expects 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norms
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data
train_data = datasets.ImageFolder('asplit_it/train_data_sorted', transform=train_transform)
test_data = datasets.ImageFolder('asplit_it/test_data_sorted', transform=test_transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

cat_b = train_data.classes
num_classes = len(cat_b)
print(f"Number of classes: {num_classes}")

# Swish activation (retained for potential customization)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Load pretrained EfficientNet-B0
model = models.efficientnet_b0(weights='IMAGENET1K_V1')  # Pretrained weights

# Freeze base layers (initially)
for param in model.parameters():
    param.requires_grad = False

# Modify classifier
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features, num_classes)
)

# Move model to device
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# Training loop
best_acc = 0.0
for epoch in range(1, 61):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f"Epoch {epoch:02}, Loss: {running_loss:.4f}, Val Acc: {acc*100:.2f}%")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_efficientnet_b0.pth')
        print("✅ Best model saved.")

    scheduler.step()

    # Unfreeze layers after 10 epochs for fine-tuning
    if epoch == 10:
        print("Unfreezing base layers for fine-tuning...")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

print(f"🎯 Final Best Accuracy: {best_acc*100:.2f}%")