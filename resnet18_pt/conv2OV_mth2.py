import torch
import torchvision.models as models
import os
import subprocess

# 8. Exportar a ONNX
#num_classes = len(train_dataset.classes)
num_classes = 26
model_onnx = models.resnet18(weights=None)
model_onnx.fc = torch.nn.Linear(model_onnx.fc.in_features, num_classes)
model_onnx.load_state_dict(torch.load("saved_model_pth/modelo_resnet18_nuevo.pth", map_location="cpu"))
model_onnx.eval()

dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "modelo_resnet18.onnx"

torch.onnx.export(
    model_onnx,
    dummy_input,
    onnx_path,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
print(f"Exportado a ONNX: {onnx_path}")

# 9. Convertir a OpenVINO usando ovc
output_dir = "openvino_model_mth2"
os.makedirs(output_dir, exist_ok=True)

try:
    cmd = ["ovc", onnx_path, "--compress_to_fp16=False", "--output_model", f"{output_dir}/modelo_resnet18"]
    print("Ejecutando:", " ".join(cmd))
    subprocess.run(cmd, check=True)
except (subprocess.CalledProcessError, FileNotFoundError):

    try:
        # Fallback a mo
        cmd = ["mo", "--input_model", onnx_path, "--compress_to_fp16", "False", "--output_dir", output_dir]
        print("Ejecutando:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error en conversión: {e}")
        print("Instala OpenVINO tools: pip install openvino-dev")

print(f"Modelo convertido a OpenVINO (XML y BIN) en: {output_dir}")