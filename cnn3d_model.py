import torch
import torch.nn as nn

class CNN3DNowcast(nn.Module):
    def __init__(self):
        super(CNN3DNowcast, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 1, kernel_size=(3, 3, 3), padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)  # (B, 64, 13, 384, 384)
        x = self.decoder(x)  # (B, 1, 13, 384, 384)
        return x[:, :, 1:]   # Output 12 frames