import torchvision.utils as vutils
import torch
def save_sample(masked, pred, real, path):
    grid = torch.cat([masked, pred, real], dim=0)
    vutils.save_image(grid, path, nrow=masked.size(0))