from flautim.pytorch.Dataset import Dataset 
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import copy

class MNISTDataset(Dataset):

    def __init__(self, data_path, **kwargs):
        super(MNISTDataset, self).__init__("MNIST", **kwargs)
        
        if isinstance(data_path, list):
            mnist = np.load(data_path[0])
            self.images = mnist['x']
            self.labels = mnist['y']

            for ct in range(1,len(data_path)):
                tmp = np.load(data_path[ct])
                self.images = np.vstack((self.images, tmp['x']))
                self.labels = np.vstack((self.labels, tmp['y']))
        else:
            mnist = np.load(data_path)
            self.images = mnist['x']
            self.labels = mnist['y']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def train(self) -> Dataset:
        return copy.deepcopy(self)

    def validation(self) -> Dataset:
        return copy.deepcopy(self) 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #img = self.transform(self.images[idx])
        # torch.Tensor(self.labels[idx
        return self.transform(self.images[idx]), torch.LongTensor([self.labels[idx]])
    
   
    

