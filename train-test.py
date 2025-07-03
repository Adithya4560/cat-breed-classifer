import torch
from torchvision import datasets, transforms
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

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4848, 0.4434, 0.4022], std=[0.2253, 0.2238, 0.2254])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4848, 0.4434, 0.4022], std=[0.2253, 0.2238, 0.2254])
])

# Load data
train_data = datasets.ImageFolder('asplit_it/train_data_sorted', transform=train_transform)
test_data = datasets.ImageFolder('asplit_it/test_data_sorted', transform=test_transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

cat_b = train_data.classes
num_classes = len(cat_b)

# Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# CNN Model
class CatNN(nn.Module):
    def __init__(self, num_classes):
        super(CatNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bat1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bat2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bat3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bat4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bat5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bat6 = nn.BatchNorm2d(1024)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(1024, 512)
        self.drop = nn.Dropout(0.5)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, num_classes)
        self.swish = Swish()

    def forward(self, x):
        x = self.pool(self.swish(self.bat1(self.conv1(x))))
        x = self.pool(self.swish(self.bat2(self.conv2(x))))
        x = self.pool(self.swish(self.bat3(self.conv3(x))))
        x = self.pool(self.swish(self.bat4(self.conv4(x))))
        x = self.pool(self.swish(self.bat5(self.conv5(x))))
        x = self.pool(self.swish(self.bat6(self.conv6(x))))
        x = self.gap(x).view(x.size(0), -1)
        x = self.drop(self.swish(self.l1(x)))
        x = self.drop(self.swish(self.l2(x)))
        return self.l3(x)

# Initialize model
model = CatNN(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

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

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_custom_cat_cnn.pth')
        print("✅ Best model saved.")

    scheduler.step()

print(f"🎯 Final Best Accuracy: {best_acc*100:.2f}%")
