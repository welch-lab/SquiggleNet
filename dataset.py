import torch
import glob
from torch.utils.data.dataset import Dataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self, zmFile, hlFile):
        z = None
        h = None
        for file in glob.glob(zmFile + '/*.pt'):
            nz = torch.load(file)
            if z is None:
                z = nz
            else:
                z = torch.cat((z, nz))
        for file in glob.glob(hlFile + '/*.pt'):
            nh = torch.load(file)
            if h is None:
                h = nh
            else:
                h = torch.cat((h, nh))
        self.data = torch.cat((z, h))
        self.label = torch.cat((torch.zeros(z.shape[0]), torch.ones(h.shape[0]))) #human: 1, others: 0

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.label[index]
        return X, y
