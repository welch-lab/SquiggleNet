import torch
from torch.utils.data.dataset import Dataset

class Dataset(torch.utils.data.Dataset):
  def __init__(self, zmFile, hlFile):
        z = torch.load(zmFile)
        h = torch.load(hlFile)
        self.data = torch.cat((z, h))
        self.label = torch.cat((torch.zeros(z.shape[0]), torch.ones(h.shape[0]))) #human: 1, others: 0

  def __len__(self):
        return len(self.label)

  def __getitem__(self, index):
        X = self.data[index]
        y = self.label[index]
        return X, y
