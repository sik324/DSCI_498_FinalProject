"""
cGAN Training Loop
Trains Generator and Discriminator for wind field super-resolution
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WindDataset


def weights_init(m):
    """Initialize weights for stable GAN training"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_cgan(G, D, X, Y, C, X_max,
               device, n_epochs=100, batch_size=16,
               lr=2e-5, lambda_l1=100.0, save_dir=None):
    """
    Train cGAN for wind field super-resolution.

    Parameters
    ----------
    G : Generator
        U-Net generator model
    D : Discriminator
        PatchGAN discriminator model
    X, Y, C : np.ndarray
        Normalized coarse, fine, condition arrays
    X_max : float
        Normalization factor
    device : torch.device
        CPU or CUDA
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate for Adam optimizer
    lambda_l1 : float
        Weight for L1 pixel loss
    save_dir : str
        Directory to save model weights

    Returns
    -------
    G : Generator
        Trained generator
    history : dict
        Training loss history
    """
    import random

    # Apply weight initialization
    G.apply(weights_init)
    D.apply(weights_init)

    # Loss functions
    criterion_adv = nn.BCEWithLogitsLoss()
    criterion_l1  = nn.L1Loss()

    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Train/val split (80/10/10)
    n       = len(X)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)
    indices = list(range(n))
    random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train+n_val]

    train_loader = DataLoader(
        WindDataset(X[train_idx], Y[train_idx], C[train_idx]),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        WindDataset(X[val_idx], Y[val_idx], C[val_idx]),
        batch_size=batch_size, shuffle=False
    )

    best_val_loss = float('inf')
    best_epoch    = 1
    history = {
        'epoch': [], 'g_loss': [], 'd_loss': [], 'val_loss': []
    }

    print(f"\nTraining on {device} — {n_epochs} epochs")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    print("=" * 60)

    for epoch in range(n_epochs):
        G.train(); D.train()
        eg = ed = 0

        for coarse, fine, cond in train_loader:
            coarse = coarse.to(device)
            fine   = fine.to(device)
            cond   = cond.to(device)

            # Train Discriminator
            opt_D.zero_grad()
            with torch.no_grad():
                fake_fine = G(coarse, cond)
            real_score = D(coarse, fine)
            fake_score = D(coarse, fake_fine)
            d_loss = (criterion_adv(real_score,
                                    torch.ones_like(real_score)) +
                      criterion_adv(fake_score,
                                    torch.zeros_like(fake_score))) * 0.5
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_D.step()

            # Train Generator
            opt_G.zero_grad()
            fake_fine  = G(coarse, cond)
            fake_score = D(coarse, fake_fine)
            adv_loss   = criterion_adv(fake_score,
                                       torch.ones_like(fake_score))
            l1_loss    = criterion_l1(fake_fine, fine)
            g_loss     = adv_loss + lambda_l1 * l1_loss
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

            eg += g_loss.item()
            ed += d_loss.item()

        nb = len(train_loader)
        eg /= nb; ed /= nb

        # Validation
        G.eval()
        val_loss = 0; n_valid = 0
        with torch.no_grad():
            for coarse, fine, cond in val_loader:
                fake = G(coarse.to(device), cond.to(device))
                v    = criterion_l1(fake, fine.to(device)).item()
                if not np.isnan(v) and not np.isinf(v):
                    val_loss += v; n_valid += 1
        val_loss = val_loss / n_valid if n_valid > 0 else float('inf')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch + 1
            if save_dir:
                torch.save(G.state_dict(),
                           os.path.join(save_dir,
                                        'generator_balanced_best.pth'))
                torch.save(D.state_dict(),
                           os.path.join(save_dir,
                                        'discriminator_balanced_best.pth'))

        history['epoch'].append(epoch + 1)
        history['g_loss'].append(round(eg, 4))
        history['d_loss'].append(round(ed, 4))
        history['val_loss'].append(round(val_loss, 4))

        print(f"Ep {epoch+1:3d}/{n_epochs} | "
              f"G:{eg:.4f} | D:{ed:.4f} | "
              f"Val:{val_loss:.4f} | "
              f"Best:{best_val_loss:.4f}(ep{best_epoch})")

    print(f"\nTraining complete!")
    print(f"Best epoch : {best_epoch}")
    print(f"Best val   : {best_val_loss:.4f}")

    # Load best weights
    if save_dir and os.path.exists(
            os.path.join(save_dir, 'generator_balanced_best.pth')):
        G.load_state_dict(torch.load(
            os.path.join(save_dir, 'generator_balanced_best.pth'),
            map_location=device))

    return G, history
