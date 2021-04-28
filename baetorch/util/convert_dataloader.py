
from torch.utils.data import Dataset, DataLoader

import numpy as np


class SimpleDataset(Dataset):
    """
    #This converts any numpy array which can fit into memory, into a pytorch dataloader instantly.
    #This use case is required for the lr range finder to work
    """

    def __init__(self, x,y=None):
        self.x=x
        self.y=y
        if y is None:
            self.y = np.arange(self.x.shape[0])
            self.y_enabled = False
        else:
            self.y_enabled =True

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx].astype(np.float32)
        y = self.y[idx].astype(np.float32)
        return x, y

def convert_dataloader(x, y=None, batch_size=100, shuffle=False):
    return DataLoader(SimpleDataset(x,y))
