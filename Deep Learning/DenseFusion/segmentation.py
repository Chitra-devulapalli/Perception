import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cv2

class SegmentationNet(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # [B, 3, H, W] -> [B, 64, H/2, W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 64, H/2, W/2] -> [B, 128, H/4, W/4]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [B, 128, H/4, W/4] -> [B, 256, H/8, W/8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # [B, 256, H/8, W/8] -> [B, 512, H/16, W/16]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 512, H/16, W/16] -> [B, 256, H/8, W/8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 256, H/8, W/8] -> [B, 128, H/4, W/4]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 128, H/4, W/4] -> [B, 64, H/2, W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, H/2, W/2] -> [B, num_classes, H, W]
            nn.Softmax(dim=1)  # Apply softmax to get class probabilities
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x