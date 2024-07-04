import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cv2
from pointnet import PointNet

class ColorEmbeddingNet(nn.Module):
    def __init__(self, drgb):
        super(ColorEmbeddingNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, drgb, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(drgb),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x