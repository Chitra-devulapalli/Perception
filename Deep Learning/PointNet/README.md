# Point Cloud Processing and Segmentation using PointNet
## Overview
This repository contains an implementation of PointNet for point cloud classification and segmentation tasks. The project explores various deep learning techniques for 3D point cloud processing, focusing on encoding spatial features, learning robust point-wise representations, and performing accurate segmentation.

The work includes:
* Implementing a dataset loader for LiDAR-based point cloud data.
* Constructing and training a PointNet-based neural network for segmentation.
* Visualizing and analyzing segmentation performance.
* Exploring improvements such as adding a T-Net transformation module for enhanced spatial understanding.


## Features
### 1. Point Cloud Data Loader
* A custom PointLoader class efficiently loads and preprocesses LiDAR point cloud data.
* Supports different training splits and label mappings.
* Implements random downsampling for efficient processing.
* Provides batch-wise collation for efficient training.
### 2. PointNet Model Implementation
* Implemented PointNet Encoder to extract hierarchical feature representations.
* Developed PointNet Segmentation Module that combines local and global features.
* Added a T-Net module to learn spatial transformations and improve segmentation accuracy.
* Designed a PointNetFull model with enhanced global feature aggregation.
### 3. Training and Evaluation
* Implemented a robust training pipeline using PyTorch.
* Evaluated segmentation performance using Intersection-over-Union (IoU) and mean IoU (mIoU) metrics.
* Conducted experiments with different hyperparameters and network configurations.
* Visualized segmentation results to assess model performance.
### 4. Visualization
* Provided visualization tools to render segmented point clouds.
* Used Plotly 3D scatter plots for interactive visualization of results.

![newplot](https://github.com/user-attachments/assets/a4be47c3-59a7-4537-85d4-32e9b5e5dc48)
