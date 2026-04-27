"""
PatchGAN Discriminator for cGAN wind field super-resolution
Architecture: Pix2Pix (Isola et al., 2017)
Input:  coarse wind field + fine wind field (real or fake)
Output: 23×23 grid of real/fake patch scores
        Each score corresponds to a 70×70 pixel receptive field
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Upsample coarse to match fine resolution
        self.coarse_up = nn.Upsample(
            size=(201, 201), mode='bilinear', align_corners=True
        )

        # PatchGAN layers
        # Input: 2 channels (coarse upsampled + fine)
        # Output: 23×23 patch scores
        self.model = nn.Sequential(
            # 201×201 → 100×100
            nn.Conv2d(2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 100×100 → 50×50
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 50×50 → 25×25
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 25×25 → 23×23
            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 23×23 → 23×23 (patch scores)
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
            # No sigmoid — use BCEWithLogitsLoss
        )

    def forward(self, coarse, fine):
        # Upsample coarse to same resolution as fine
        coarse_up = self.coarse_up(coarse)
        # Concatenate as 2-channel input
        x = torch.cat([coarse_up, fine], dim=1)
        return self.model(x)  # (B, 1, 23, 23)
