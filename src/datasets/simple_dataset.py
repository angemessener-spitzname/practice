import torch
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, image, mask):
        '''
        ???????????????????
            image : list of ndarray
            mask : list of ndarray
        '''
        self.image = image
        self.mask = mask

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = self.image[idx]
        msk = self.mask[idx]
        sample = {
                    'image': torch.from_numpy(img).type(torch.FloatTensor),
                    'mask': torch.from_numpy(msk).type(torch.FloatTensor)
                  }
        return sample