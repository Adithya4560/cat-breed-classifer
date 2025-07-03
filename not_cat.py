import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. Data transforms for training and validation
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# 2. Prepare datasets (example assumes folders: train/cat, train/notcat, val/cat, val/notcat)
train_dataset = datasets.ImageFolder(r"C:\Users\Lenovo\OneDrive\Desktop\cat project\dataset\train", transform=train_transforms)
val_dataset = datasets.ImageFolder(r"C:\Users\Lenovo\OneDrive\Desktop\cat project\dataset\val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 3. Load pretrained MobileNetV2 and modify last layer
model = models.mobilenet_v2(pretrained=True)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)  # Binary output: cat or not-cat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 5. Training loop (simplified)
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = correct_train / total_train

    # 🔎 Validation accuracy
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = correct_val / total_val

    print(f"Epoch {epoch+1} | Train Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_accuracy:.2f} | Val Loss: {val_loss/len(val_loader):.4f} | "
          f"Val Acc: {val_accuracy:.2f}")

# 6. Save the trained model
torch.save(model.state_dict(), "cat_notcat_model.pth")
