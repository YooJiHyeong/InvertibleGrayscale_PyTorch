import torch.nn as nn
from modules import FlatConv, ResidualBlock, UpConv, DownConv


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            FlatConv(3, 64, "relu"),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        self.down = nn.Sequential(
            DownConv(64,  128),
            DownConv(128, 256)
        )

        self.bridge = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        self.up = nn.Sequential(
            UpConv(256, 128),
            UpConv(128, 64),
        )

        self.tail = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            FlatConv(64, 1, "tanh")
        )

    def forward(self, x):
        out = self.head(x)

        skip = []
        for d in self.down:
            skip.append(out)
            out = d(out)

        out = self.bridge(out)

        for u, s in zip(self.up, skip[::-1]):
            out = u(out)
            out = out + s

        out = self.tail(out)

        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            FlatConv(64, 3, 'tanh')
        )

    def forward(self, x):
        return self.layers(x)
