import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

from densefusion import DenseFusionNet

def pose_loss(pred_pose, gt_pose, confidence):
    loss = torch.nn.functional.mse_loss(pred_pose, gt_pose) * confidence
    return loss.mean()

def train(dataloader, model, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        img, pcld, gt_pose = batch
        img, pcld, gt_pose = img.to(device), pcld.to(device), gt_pose.to(device)

        optimizer.zero_grad()

        pred_pose = model(img, pcld)
        loss = criterion(pred_pose, gt_pose)
        loss.backward()
        optimizer.step()

        print(f'Loss: {loss.item()}')

class PoseEstimationDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, pose_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.pose_dir = pose_dir
        self.transform = transform
        self.rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
        self.depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
        self.pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        pose_path = os.path.join(self.pose_dir, self.pose_files[idx])

        rgb_image = cv2.imread(rgb_path)
        depth_map = np.load(depth_path)
        pose = np.load(pose_path)

        # Convert depth map to point cloud (assuming depth_map is HxW and pose is [R|t])
        point_cloud = self.depth_to_point_cloud(depth_map, pose)

        if self.transform:
            rgb_image = self.transform(rgb_image)

        # Normalize point cloud and convert to tensor
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        rgb_image = torch.tensor(rgb_image.transpose((2, 0, 1)), dtype=torch.float32) / 255.0
        pose = torch.tensor(pose, dtype=torch.float32)

        return rgb_image, point_cloud, pose

    def depth_to_point_cloud(self, depth_map, pose):
        fx, fy = 1.0, 1.0  # Replace with actual focal lengths
        cx, cy = depth_map.shape[1] // 2, depth_map.shape[0] // 2
        points = []
        for v in range(depth_map.shape[0]):
            for u in range(depth_map.shape[1]):
                z = depth_map[v, u]
                if z > 0:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append([x, y, z])
        points = np.dot(np.array(points), pose[:3, :3].T) + pose[:3, 3]
        return points

rgb_dir = 'path_to_rgb_images'
depth_dir = 'path_to_depth_maps'
pose_dir = 'path_to_ground_truth_poses'

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseFusionNet(drgb=128, dgeo=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = pose_loss

    dataset = PoseEstimationDataset(rgb_dir, depth_dir, pose_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    num_epochs = 20
    # Train the model
    for epoch in range(num_epochs):
        train(dataloader, model, optimizer, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs} completed')

    # Save the model
    torch.save(model.state_dict(), 'densefusion_model.pth')

if __name__ == "__main__":
    main()