import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from numpy import asarray

class SegmentationDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str) -> None:
        images = []
        for root, _, fnames in os.walk(image_dir):
            for fname in fnames:
                images.append(os.path.join(root, fname))

        masks = []
        for root, _, fnames in os.walk(mask_dir):
            for fname in fnames:
                masks.append(os.path.join(root, fname))               
                
        self.image_list = images
        self.mask_list = masks

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> dict:
        img = asarray(Image.open(self.image_list[idx]))
        msk = asarray(Image.open(self.mask_list[idx]))
        sample = {
                    'image': torch.from_numpy(img).type(torch.FloatTensor),
                    'mask': torch.from_numpy(msk).type(torch.FloatTensor)
                  }
        return sample

