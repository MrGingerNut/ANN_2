import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json

# ── Cargar clases guardadas durante entrenamiento ──────────────────────────────
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)
print(f"Clases cargadas ({NUM_CLASSES}): {CLASS_NAMES}")

# ── Configuración ──────────────────────────────────────────────────────────────
MODEL_PATH = "best_model_resnet50.pth"
IMAGE_DIR  = "/home/liese2/ANN_project/resnet50_pt/Predict/"

# ── Transformaciones ───────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Cargar modelo ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ── Inferencia ─────────────────────────────────────────────────────────────────
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.png')]

if not image_files:
    print("No se encontraron imágenes BMP.")
    exit()

print(f"\nProcesando {len(image_files)} imágenes...\n")
print(f"{'Archivo':<40} {'Real':<8} {'Predicción':<12} {'Confianza':>10}  OK")
print("-" * 80)

correct = 0
total   = 0

for filename in sorted(image_files):
    true_label = filename[0].upper()  # primera letra del nombre = clase real

    image  = Image.open(os.path.join(IMAGE_DIR, filename)).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    predicted_label = CLASS_NAMES[pred_idx.item()]
    confidence_pct  = confidence.item() * 100
    match           = "✓" if predicted_label == true_label else "✗"

    print(f"{filename:<40} {true_label:<8} {predicted_label:<12} {confidence_pct:>8.2f}%  {match}")

    correct += predicted_label == true_label
    total   += 1

# ── Resumen ────────────────────────────────────────────────────────────────────
print("-" * 80)
print(f"\nTotal: {total} | Correctas: {correct} | Accuracy: {100 * correct / total:.2f}%")