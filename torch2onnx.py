import torch
from src.model import DenoisingAutoencoder

# Load the model
model_path = 'best_dae_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenoisingAutoencoder(input_channels=3).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Export the model to ONNX
dummy_input = torch.randn(1, 3, 256, 256).to(device)
torch.onnx.export(model=model, args=dummy_input, f='dae_model.onnx')
print('Model exported to dae_model.onnx')
