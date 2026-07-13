import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

DATASET_PATH = "dataset"
BATCH_SIZE = 16
EPOCHS = 10
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# TRANSFORMS
# =========================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# =========================
# DATASET LOADER
# =========================

train_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_data, val_data = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=BATCH_SIZE
)

print("Classes:", train_dataset.classes)

# =========================
# LOAD EFFICIENTNET
# =========================

model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 2)

model = model.to(device)

# =========================
# LOSS & OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAIN LOOP
# =========================

train_acc_list = []
val_acc_list = []

for epoch in range(EPOCHS):
    model.train()
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    train_acc_list.append(train_acc)

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(), "coral_efficientnet.pth")
print("✅ Model Saved!")

# =========================
# ACCURACY GRAPH
# =========================

plt.plot(train_acc_list)
plt.plot(val_acc_list)
plt.title("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()
