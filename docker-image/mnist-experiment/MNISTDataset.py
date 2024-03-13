from programming_api.Dataset import Dataset 
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import copy

class MNISTDataset(Dataset):

    def __init__(self, data_path, **kwargs):
        super(MNISTDataset, self).__init__("MNIST", **kwargs)
        
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
    
   
    

