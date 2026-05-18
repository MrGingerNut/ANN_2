import torch
import torchvision.models as models
num_classes = 26

# Cargar el modelo ResNet-18 entrenado
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("modelo_resnet18_nuevo.pth"))
model.eval()

# Crear un tensor de entrada de ejemplo para la exportación
dummy_input = torch.randn(1, 3, 224, 224)  # batch_size=1, channels=3 (RGB), height=224, width=224

# Exportar a ONNX
torch.onnx.export(model, dummy_input, "modelo_resnet18.onnx", opset_version=11)

#from openvino.tools.mo import convert_model
# Convertir el modelo ONNX a OpenVINO IR
#convert_model("modelo_resnet18.onnx", output_dir="./openvino_model")

from openvino.runtime import serialize
from openvino.tools.mo import convert_model

# Ruta del modelo de entrada y carpeta de salida
input_model = "modelo_resnet18.onnx"
output_dir = "./openvino_model"

# Convertir el modelo y obtenerlo como objeto de OpenVINO
model = convert_model(input_model)

import os

output_dir = "openvino_model_mth1"
os.makedirs(output_dir, exist_ok=True)  # Crea el directorio si no existe

# Guardar el modelo en formato IR (.xml y .bin)
serialize(model, f"{output_dir}/modelo_resnet18.xml", f"{output_dir}/modelo_resnet18.bin")

print(f"Modelo convertido y guardado en: {output_dir}")