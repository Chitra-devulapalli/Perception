import torch
from torch import nn
import torch.nn.functional as F



def positional_encoding(x, num_embed):
    """
    Apply positional encoding to the input tensor.

    Args:
        x (Tensor): The input tensor to be positionally encoded. shape: (N,3)
        num_embed (int): The number of encoding frequencies for sine and cosine.

    Returns:
        encodings (Tensor): The input tensor augmented with positional 
                            encodings. shape: (N, 3 + 3 * 2 * num_embed)
    """
    encoding = None
    ######################################################################
    # TODO: positional_encoding funtion
    # fill in as described in the notebook
    ######################################################################
    encoding = [x]
    for i in range(num_embed):
        scale = 2 ** i
        encoding.append(torch.sin(x * scale))
        encoding.append(torch.cos(x * scale))
    encoding = torch.cat(encoding, dim=1)

    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    return encoding


def get_rays(height, width, focal, cam_to_world):
    """
    Generate rays for each pixel in the image plane.

    Args:
    - height (int): The height of the image in pixels.
    - width (int): The width of the image in pixels.
    - focal (float): The focal length of the camera.
    - cam_to_world (Tensor): A 4x4 camera-to-world transformation matrix.

    Returns:
    - rays_o (Tensor): Origins of the rays in world space. Shape: (H, W, 3)
    - rays_d (Tensor): Directions of the rays in world space. Shape: (H, W, 3)
    """
    rays_o = None
    rays_d = None
    ######################################################################
    # TODO: get_rays funtion    
    # 1. A tensor of all the pixel coordinates
    # 2. Normalize pixel coordinates to camera space
    # 3. Transform ray directions from camera space to world space
    # 4. Set ray origins
    # Hint: Do this without any python loops, functions that may help: 
    # torch.meshgrid, torch.stack, torch.sum, torch.expand_as, torch.einsum
    # there are may ways to do this you do not need to use any of the suggestions     
    ######################################################################
    i = torch.arange(width, dtype=torch.float32).to(cam_to_world.device)
    j = torch.arange(height, dtype=torch.float32).to(cam_to_world.device)

    i, j = torch.meshgrid(i, j, indexing='ij')
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)

    x = (i - width * .5) / focal
    y = (j - height * .5) / focal

    directions = torch.stack([x, -y, -torch.ones_like(i)], dim=-1)

    rays_d = torch.sum(directions[..., None, :] * cam_to_world[:3, :3], dim=-1)

    rays_o = cam_to_world[:3, -1].expand(rays_d.shape)


    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    return rays_o, rays_d




class nerf(nn.Module):
    def __init__(self, depth=8, width=256, num_embed=0):
        """
        Initialize the NeRF model.

        Args:
            depth (int): Number of layers in the model.
            width (int): Number of neurons in each hidden layer.
            num_embed (int): Number of frequency bands for positional encoding.
        """
        super(nerf, self).__init__()
        self.depth = depth
        self.num_embed = num_embed
        ######################################################################
        # TODO: initallize all the layers
        ######################################################################    
        layers_list = [torch.nn.Linear(3 + 3 * 2 * self.num_embed, width)]

        for i in range(self.depth - 1):
            if i % 4 == 0:
                layers_list.append(torch.nn.Linear(width + (3 + 3 * 2 * self.num_embed), width))
            else:
                layers_list.append(torch.nn.Linear(width, width))

        self.layers = torch.nn.ModuleList(layers_list)
        self.output_layer = torch.nn.Linear(width, 4)

        #################################################################
        #                        END OF YOUR CODE                       #
        #################################################################

    def forward(self, x):
        output = None
        ######################################################################
        # TODO: Pass the input through the layers, remembering to use the 
        # original input at every 4th layer after the first
        ######################################################################    
        x_orig = x
        
        for i, layer in enumerate(self.layers):
            x = torch.nn.functional.relu(layer(x))
            if i % 4 == 0:
                x = torch.cat([x, x_orig], dim=-1)
        
        output = self.output_layer(x)
        #################################################################
        #                        END OF YOUR CODE                       #
        #################################################################

        return output

def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):

    def batchify(fn, chunk=1024*32):
        def batch_fn(inputs):
            results = []
            for i in range(0, inputs.shape[0], chunk):
                results.append(fn(inputs[i:i+chunk]))
            return torch.cat(results, 0)
        return batch_fn

    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples, device=rays_o.device)
    if rand:
        z_vals = z_vals + torch.rand(list(rays_o.shape[:-1]) + [N_samples], device=rays_o.device) * (far - near) / N_samples
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Run network
    pts_flat = pts.reshape(-1, 3)
    pts_flat = positional_encoding(pts_flat, network_fn.num_embed) 
    raw = batchify(network_fn)(pts_flat)
    raw = raw.reshape(list(pts.shape[:-1]) + [4])

    # Compute opacities and colors
    sigma_a = F.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3])

    # Do volume rendering
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.broadcast_to(torch.tensor([1e10], device=rays_o.device), z_vals[..., :1].shape)], -1)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], -1), -1)[:, :, :-1]

    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    return rgb_map, depth_map, acc_map


