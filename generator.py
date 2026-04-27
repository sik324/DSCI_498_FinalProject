"""
U-Net Generator for cGAN wind field super-resolution
Architecture: Pix2Pix (Isola et al., 2017)
Input:  coarse wind field (1, 22, 21) + condition vector (4,)
Output: fine wind field (1, 201, 201)
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, condition_dim=4):
        super().__init__()

        # Condition embedding — maps storm params to spatial map
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 22 * 21),
            nn.ReLU()
        )

        # Encoder — feature extraction
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Decoder with skip connections
        self.dec3 = nn.Sequential(
            nn.Conv2d(768, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(384, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(192, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Final upsampling 22×21 → 201×201
        self.upsample = nn.Sequential(
            nn.Upsample(size=(201, 201), mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, condition):
        # Embed condition vector into spatial map
        cond = self.condition_embed(condition)
        cond = cond.view(-1, 1, 22, 21)

        # Concatenate wind field with condition
        x  = torch.cat([x, cond], dim=1)  # (B, 2, 22, 21)

        # Encoder
        e1 = self.enc1(x)                 # (B, 64, 22, 21)
        e2 = self.enc2(e1)                # (B, 128, 22, 21)
        e3 = self.enc3(e2)                # (B, 256, 22, 21)

        # Bottleneck
        b  = self.bottleneck(e3)          # (B, 512, 22, 21)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([b,  e3], dim=1))  # (B, 256)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # (B, 128)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # (B, 64)

        # Upsample to fine resolution
        return self.upsample(d1)          # (B, 1, 201, 201)
