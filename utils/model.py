import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.pixel_shuffle(self.conv(x)))
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(SelfAttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class Model(nn.Module):
    def __init__(self, upscale_factor, embedding_size=64, dropout_p=0.1, weight_decay=0.1):
        super(Model, self).__init__()
        self.upscale_factor = upscale_factor
        self.embedding = nn.Embedding(256, embedding_size)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.residual1 = ResidualBlock(64)
        self.residual2 = ResidualBlock(64)
        self.self_attention = SelfAttentionBlock(64)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU()
        )
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        residual = x
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.self_attention(x)
        x += residual
        x = self.upsample(x)
        x = self.conv2(x)
        return x

    def l2_regularization_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.weight_decay * l2_loss
