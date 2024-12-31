import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn
import collections

collections.Iterable = collections.abc.Iterable

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        hw = height * width  # Total spatial positions

        # Query and Key
        query = self.query_conv(x).view(batch_size, -1, hw).permute(0, 2, 1)  # [B, HW, C']
        key = self.key_conv(x).view(batch_size, -1, hw)  # [B, C', HW]

        # Compute Attention
        attention = torch.bmm(query, key)  # [B, HW, HW]
        attention = self.softmax(attention / (channels ** 0.5))  # Normalized attention map

        # Value
        value = self.value_conv(x).view(batch_size, -1, hw)  # [B, C, HW]
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, HW]

        # Reshape back to spatial dimensions
        out = out.view(batch_size, channels, height, width)
        return out + x  # Residual connection


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # 256x256 -> 128x128
            nn.ReLU(),
            ResidualBlock(32)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),              # 128x128 -> 64x64
            nn.ReLU(),
            ResidualBlock(64)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),             # 64x64 -> 32x32
            nn.ReLU(),
            ResidualBlock(128)
        )

        # Decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # 32x32 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            AttentionBlock(64)  # Tambah Attention
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # 64x64 -> 128x128
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            AttentionBlock(32)  # Tambah Attention
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x128 -> 256x256
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        # Decoder with attention-enhanced skip connections
        dec3 = self.decoder3(enc3)
        dec2 = self.decoder2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.decoder1(torch.cat([dec2, enc1], dim=1))
        
        return dec1


# Load the model
model_path = 'best_dae_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenoisingAutoencoder(input_channels=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Define the transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load and process the input image
input_image_path = 'images.jpeg'
input_image = Image.open(input_image_path).convert('RGB')

# get original image size
original_size = input_image.size
input_tensor = transform(input_image).unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    denoised_tensor = model(input_tensor)

# Convert the output tensor to an image
denoised_image = denoised_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
denoised_image = (denoised_image * 255).astype(np.uint8)
denoised_image = Image.fromarray(denoised_image)

# Save the denoised image and convert it back to the original size
denoised_image = denoised_image.resize(original_size)
output_image_path = 'result.jpeg'
denoised_image.save(output_image_path)

print(f'Denoised image saved to {output_image_path}')