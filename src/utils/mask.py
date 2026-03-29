import torch
import random

def random_mask(img):
    _, h, w = img.shape
    mask = torch.ones((1, h, w))

    mask_size = random.randint(20, 50)
    x = random.randint(0, w - mask_size)
    y = random.randint(0, h - mask_size)

    mask[:, y:y+mask_size, x:x+mask_size] = 0
    return mask