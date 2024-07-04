import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cv2
from pointnet import PointNet

class GeometryEmbeddingNet(PointNet):
    def __init__(self, dgeo):
        super(GeometryEmbeddingNet, self).__init__()
        self.dgeo = dgeo
    
    def forward(self, x):
        return super(GeometryEmbeddingNet, self).forward(x)