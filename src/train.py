import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# -------------------------
# Configuration
# -------------------------
DATA_DIR = "data/raw/images"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Utilisation de : {DEVICE}")

# -------------------------
# Augmentations et transforms
# -------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# Chargement du dataset
# -------------------------
print("Chargement du dataset...")
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
print(f"Classes détectées : {full_dataset.classes}")
print(f"Total images : {len(full_dataset)}")

# -------------------------
# Séparation train/val/test
# -------------------------
total = len(full_dataset)
train_size = int(0.70 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size

train_set, val_set, test_set = random_split(
    full_dataset, [train_size, val_size, test_size]
)

# Appliquer val_transforms sur val et test
val_set.dataset.transform = val_transforms
test_set.dataset.transform = val_transforms

print(f"Train : {train_size} | Val : {val_size} | Test : {test_size}")

# -------------------------
# DataLoaders
# -------------------------
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Modèle ResNet-18
# -------------------------
print("Chargement du modèle ResNet-18...")
model = models.resnet18(weights="IMAGENET1K_V1")

num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# -------------------------
# Loss et optimiseur
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------
# Entraînement
# -------------------------
best_val_acc = 0.0

print("\nDébut de l'entraînement...")
for epoch in range(NUM_EPOCHS):
    # -- Phase entraînement --
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()

    train_acc = train_correct / train_size

    # -- Phase validation --
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()

    val_acc = val_correct / val_size

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | "
          f"Val Acc: {val_acc:.4f}")

    # -- Sauvegarde du meilleur modèle --
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✔ Meilleur modèle sauvegardé (val_acc={val_acc:.4f})")

# -------------------------
# Evaluation finale sur test
# -------------------------
print("\nEvaluation finale sur le jeu de test...")
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

test_correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()

test_acc = test_correct / test_size
print(f"Test Accuracy finale : {test_acc:.4f}")
print(f"\n✔ Modèle sauvegardé dans : {MODEL_PATH}")