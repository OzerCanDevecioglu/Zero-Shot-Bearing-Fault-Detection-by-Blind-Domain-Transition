import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from scipy.io import loadmat
import torch
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from Fastonn import SelfONNTranspose1d as SelfONNTranspose1dlayer
from Fastonn import SelfONN1d as SelfONN1dlayer
from utils import ECGDataset, ECGDataModule,init_weights,TECGDataset,TECGDataModule
  


  
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=False):
        super(Upsample, self).__init__()
        self.dropout = dropout
        self.block = nn.Sequential(
            SelfONNTranspose1dlayer(in_channels, out_channels, kernel_size, stride, padding, bias=nn.InstanceNorm1d,q=3),
            nn.InstanceNorm1d(out_channels),
            nn.Tanh()
        )
        self.dropout_layer = nn.Dropout(0.5)

    def forward(self, x, shortcut=None):
        x = self.block(x)
        if self.dropout:
            x = self.dropout_layer(x)

        if shortcut is not None:
            x = torch.cat([x, shortcut], dim=1)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, apply_instancenorm=False):
        super(Downsample, self).__init__()
        self.conv = SelfONN1dlayer(in_channels, out_channels, kernel_size, stride, padding, bias=nn.InstanceNorm1d,q=3)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.relu = nn.Tanh()
        self.apply_norm = apply_instancenorm
        

    def forward(self, x):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
        x = self.relu(x)

        return x

class CycleGAN_Unet_Generator(nn.Module):
    def __init__(self, filter=8):
        super(CycleGAN_Unet_Generator, self).__init__()
        self.downsamples = nn.ModuleList([
            Downsample(1, filter, kernel_size=5, padding=1,apply_instancenorm=False),  # (b, filter, 128, 128)
            Downsample(filter, filter * 2, kernel_size=5,padding=1),  # (b, filter * 2, 64, 64)
            Downsample(filter * 2, filter * 4, kernel_size=5,padding=1),  # (b, filter * 4, 32, 32)
            Downsample(filter * 4, filter * 8, kernel_size=5,padding=1),  # (b, filter * 8, 16, 16)
            Downsample(filter * 8, filter * 8, kernel_size=5,padding=1), # (b, filter * 8, 8, 8)
        ])

        self.upsamples = nn.ModuleList([
            Upsample(filter * 8, filter * 8, kernel_size=5,padding=1),
            Upsample(filter * 16, filter * 4, dropout=False, kernel_size=5,padding=1),
            Upsample(filter * 8, filter * 2, dropout=False, kernel_size=5,padding=1),
            Upsample(filter * 4, filter, dropout=False, kernel_size=5,padding=1)
        ])

        self.last = nn.Sequential(
            SelfONNTranspose1dlayer(filter * 2, 1, kernel_size=6, stride=2, padding=1,q=3),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for l in self.downsamples:
            x = l(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for l, s in zip(self.upsamples, skips):
            x = l(x, s)
        out = self.last(x)

        return out













class CycleGAN_Discriminator(nn.Module):
    def __init__(self, filter=8):
        super(CycleGAN_Discriminator, self).__init__()

        self.block = nn.Sequential(
            Downsample(1, filter, kernel_size=4, stride=4, apply_instancenorm=False),
            Downsample(filter, filter * 2, kernel_size=4, stride=4),
            Downsample(filter * 2, filter * 4, kernel_size=4, stride=4),
            Downsample(filter * 4, filter * 8, kernel_size=4, stride=4),
            Downsample(filter * 8, filter * 16, kernel_size=4, stride=4),
        )

        self.last = SelfONN1dlayer(filter * 16, 1, kernel_size=6, stride=4, padding=1,q=3)

    def forward(self, x):
        x = self.block(x)
        x = self.last(x)

        return x


model = CycleGAN_Discriminator()
x = torch.randn(1,1,  4096)

# Let's print it
out=model(x)
print(out.size())

