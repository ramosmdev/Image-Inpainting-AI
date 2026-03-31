import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import random
import os
from src.utils.mask import random_mask

import src.config as config

class InpaintingDataset(Dataset):
    def __init__(self, img_dir, size=None):
        self.img_dir = img_dir
        self.files = os.listdir(img_dir)
        
        # Use config.IMG_SIZE if no size is specified
        target_size = size if size is not None else config.IMG_SIZE
        
        self.transform = T.Compose([
            T.Resize((target_size, target_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def random_mask(self, img):
        _, h, w = img.shape
        mask = torch.ones((1, h, w))

        # máscara cuadrada random
        mask_size = random.randint(20, 50)
        x = random.randint(0, w - mask_size)
        y = random.randint(0, h - mask_size)

        mask[:, y:y+mask_size, x:x+mask_size] = 0
        return mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        mask = random_mask(img)
        masked_img = img * mask

        return masked_img, mask, img