import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# 1. Hiperparámetros
batch_size = 8
learning_rate = 0.001
num_epochs = 1

# 2. Transformaciones para el dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=r'/home/liese2/ANN_project/resnet18_pt/Dataset/Train', transform=transform)
test_dataset = datasets.ImageFolder(root=r'/home/liese2/ANN_project/resnet18_pt/Dataset/Test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Dataset ready")

# 4. Definir la ResNet-18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # Ajustar para la detección de lenguaje de señas

# 5. Configuración de entrenamiento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device es: ", device)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# 6. Función de entrenamiento
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# 7. Función de validación
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy

# 8. Entrenamiento de la red
for epoch in range(num_epochs):
    print("Starting epoch num", epoch)

    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = validate(model, test_loader, criterion, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
          f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

import openvino as ov
import os

output_dir = "openvino_model_mth3"
os.makedirs(output_dir, exist_ok=True)  # Crea el directorio si no existe

ov_model= ov.convert_model(model, input=[1, 3, 224, 224])
ov.save_model(ov_model, f"{output_dir}/modelo_resnet18.xml")

