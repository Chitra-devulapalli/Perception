import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cv2
from pointnet import PointNet

class PoseRefinementNet(nn.Module):
    def __init__(self):
        super(PoseRefinementNet, self).__init__()
        self.refine_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # Output: 3 for rotation residual, 3 for translation residual
        )
    
    def forward(self, fused_features):
        return self.refine_net(fused_features)