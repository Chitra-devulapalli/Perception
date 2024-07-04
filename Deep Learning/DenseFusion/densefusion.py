import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import cv2
from pointnet import PointNet
from geometryembedding import GeometryEmbeddingNet
from colorembedding import ColorEmbeddingNet

class DenseFusionNet(nn.Module):
    def __init__(self, drgb, dgeo, num_classes):
        super(DenseFusionNet, self).__init__()
        self.segmentation_net = SegmentationNet(num_classes)
        self.color_net = ColorEmbeddingNet(drgb)
        self.geo_net = GeometryEmbeddingNet(dgeo)
        self.fusion_net = nn.Sequential(
            nn.Conv1d(drgb + dgeo, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 7, 1)  # Output: 3 for rotation, 3 for translation, 1 for confidence
        )
    
    def forward(self, img, pcld):
        # Segment the image to get masks
        masks = self.segmentation_net(img)
        # Extract color features from the segmented image
        color_features = self.color_net(img * masks)
        # Extract geometric features from the point cloud
        geo_features = self.geo_net(pcld)
        # Concatenate color and geometric features
        fusion_features = torch.cat((color_features, geo_features), dim=1)
        # Predict initial pose and confidence
        initial_output = self.fusion_net(fusion_features)
        
        # Extract the initial pose estimation
        initial_pose = initial_output[:, :6]
        confidence = initial_output[:, 6]
        
        # Refine the pose estimation iteratively
        refined_pose = initial_pose
        for _ in range(3):  # Number of refinement iterations can be adjusted
            refined_features = torch.cat((fusion_features, refined_pose), dim=1)
            residuals = self.refine_net(refined_features)
            refined_pose += residuals
        
        return refined_pose, confidence