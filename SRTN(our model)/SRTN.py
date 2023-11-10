import torch
import torch.nn as nn
from ScConv import ScConv

class SRTN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, num_features=64, num_layers=16, num_heads=8):
        super(SRTN, self).__init__()
        self.head = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.residual_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_layers)]
        )
        self.transformer = EfficientTransformer(num_features, num_heads)
        self.upsample = UpsampleBlock(num_features, scale_factor)  # Use pixel shuffle for upsampling
        self.tail = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)

        residual_head = x  # Save residual from head
        x = self.body(x)
        x += residual_head  # Apply the first residual connection

        residual_conv = self.residual_conv(residual_head)

        residual_body = x  # Save residual from body
        x = self.transformer(x)
        x += residual_body  # Apply the second residual connection
        x += residual_conv

        x = self.upsample(x)
        x = self.tail(x)
        return x

# Input shape: torch.Size([16, 3, 64, 64])
# After head shape: torch.Size([16, 64, 64, 64])
# After body shape: torch.Size([16, 64, 64, 64])
# After transformer shape: torch.Size([16, 64, 64, 64])
# After upsample shape: torch.Size([16, 64, 256, 256])
# After tail shape: torch.Size([16, 3, 256, 256])

# Residual Block (Simple ResBlock)
class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = ScConv(num_features)
        self.relu = nn.ReLU()
        self.conv2 = ScConv(num_features)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual
        return x

# Efficient Transformer Module
class EfficientTransformer(nn.Module):
    def __init__(self, channels, heads):
        super(EfficientTransformer, self).__init__()
        self.input_embeddings = nn.Conv2d(channels, channels, kernel_size=1)
        self.transformer_block = TransformerBlock(channels, heads)
        self.output_embeddings = nn.Conv2d(channels, 64, kernel_size=1)

    def forward(self, x):
        x = self.input_embeddings(x)
        x = self.transformer_block(x)
        x = self.output_embeddings(x)
        return x

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, channels, heads):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = EMA(channels, heads)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = MLP(channels)

    def forward(self, x):
        x = ResidualAdd(nn.Sequential(self.norm1, self.attn))(x)
        x = ResidualAdd(nn.Sequential(self.norm2, self.mlp))(x)
        return x

# Residual Add Module
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

# Multi-Layer Perceptron Module
class MLP(nn.Module):
    def __init__(self, channels):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(channels, channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(channels * 4, channels)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# EMA Module (Exponential Moving Average)
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# Example usage:
channels = 512  # Example channel size
heads = 8       # Example number of heads
model = EfficientTransformer(channels, heads)


# Upsample Block Module
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
