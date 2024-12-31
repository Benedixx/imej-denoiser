import torch
from torch import nn


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
        hw = height * width  

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
