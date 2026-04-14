"""
Reconstruct images from VGG features.

Author: Pierre Lelievre
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torchvision import models
from torchvision.models import feature_extraction

VGG_IMG_SIZE = 224
VGG_MODEL = models.vgg16_bn
VGG_WEIGHTS = models.VGG16_BN_Weights.IMAGENET1K_V1
IMGNET_MEAN = (0.485, 0.456, 0.406)
IMGNET_STD = (0.229, 0.224, 0.225)


def print_vgg_layers():
    for layer in feature_extraction.get_graph_node_names(VGG_MODEL())[1]:
        print(layer)


def load_img(path):
    assert os.path.isfile(path), 'Image not found.'
    # Load image
    while True:
        try:
            img = Image.open(path).convert('RGB')
            # Resize image
            img = img.resize(
                (VGG_IMG_SIZE, VGG_IMG_SIZE),
                resample=Image.Resampling.LANCZOS)
            break
        except OSError:
            print(f'Failed loading : {path}')
    # Remap image in range [0, 1]
    img = np.array(img, dtype=np.float32)
    img /= 255.0
    return img


def clean_img(img, rescale=False):
    img = img.copy()
    if rescale:
        img -= np.min(img)
        img /= np.max(img)
    return np.clip(img, 0.0, 1.0)


class VGGRec:
    learning_rate = 1e-2
    scheduler_decay = 0.9
    scheduler_patience = 10
    scheduler_min_lr = 1e-5
    dtype = torch.float32
    def __init__(self, layer, device=None):
        # Set PyTorch default dtype
        torch.set_default_dtype(self.dtype)
        # Set device
        self.device = torch.device('cpu')
        if isinstance(device, int):
            device = f'cuda:{device}'
        if isinstance(device, str):
            if len(device) >= 4 and device[:4] == 'cuda' and (
                    torch.cuda.is_available()):
                self.device = torch.device(device)
                torch.cuda.device(self.device)
            elif device == 'mps' and torch.backends.mps.is_available():
                self.device = torch.device(device)
            elif device == 'cpu':
                self.device = torch.device(device)
            else:
                raise ValueError(f'Device {device} is not available.')
        # Load VGG
        self.layer = layer
        self.vgg = feature_extraction.create_feature_extractor(
            VGG_MODEL(weights=VGG_WEIGHTS).eval(), return_nodes=[self.layer])
        self.vgg.to(self.device)

    def _prepare_img(self, img):
        img = img.copy()
        img -= IMGNET_MEAN
        img /= IMGNET_STD
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1).unsqueeze(dim=0).to(self.device)
        return img

    def _recover_img(self, img):
        img = img.squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
        img *= IMGNET_STD
        img += IMGNET_MEAN
        return img

    def get_features(self, img):
        img = self._prepare_img(img)
        return self.vgg(img)[self.layer].detach().cpu().numpy()

    def reconstruct_img(self, target_feat, img_0=None, n_steps=100, seed=100):
        # Prepare target feature
        target_feat = torch.tensor(
            target_feat, dtype=torch.float32).to(self.device)
        # Prepare reconstructed image
        if img_0 is None:
            img_0 = np.random.default_rng(seed).normal(
                size=(VGG_IMG_SIZE, VGG_IMG_SIZE, 3))
        img = self._prepare_img(img_0).requires_grad_(True) # declair we want the gradient on image optimizing
        # Optimizer
        optimizer = optim.Adam([img], lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.scheduler_decay,
            patience=self.scheduler_patience, min_lr=self.scheduler_min_lr)
        # Loss
        mse_loss = nn.MSELoss(reduction='none')
        # Initial loss
        self.vgg.eval()
        self.vgg.zero_grad(set_to_none=True)
        feat = self.vgg(img)[self.layer]
        loss = mse_loss(feat, target_feat).mean()
        print(f'Initial loss : {loss.item():.4f}')
        # Iterate
        for _ in tqdm(range(n_steps), total=n_steps, desc='reconstruct'):
            self.vgg.zero_grad(set_to_none=True)
            feat = self.vgg(img)[self.layer]
            loss = mse_loss(feat, target_feat).mean()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
        print(f'Final loss   : {loss.item():.4f}')
        return self._recover_img(img.detach())
