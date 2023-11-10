import torch
import torch.nn as nn

class EDSR(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, num_features=64, num_layers=16):
        super(EDSR, self).__init__()
        self.head = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_layers)]
        )
        self.upsample = UpsampleBlock(num_features, scale_factor)  # Using pixel shuffle convolutional upsampling
        self.tail = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        residual = x
        x = self.body(x)
        x += residual
        x = self.upsample(x)
        x = self.tail(x)
        return x

# Residual layer (Simple ResBlock)
class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual
        return x

# Pixel shuffle convolutional layer
class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x
