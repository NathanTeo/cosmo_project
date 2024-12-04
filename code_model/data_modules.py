"""
Author: Nathan Teo

This script contains all data modules.
"""

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

class BlobDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning data module for loading and transforming the real sample gaussian blobs
    """
    def __init__(self, data_file, batch_size, num_workers, train_split=1, truncate_ratio=None, transforms=None):
        super().__init__()
        
        # Parameters
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_split = train_split
        
        self.truncate_ratio = truncate_ratio
        
        if transforms is not None:
            self.transforms = [transform_dict[transform] for transform in transforms]
        else:
            self.transforms = None
        
        self.init_attributes()
    
    def init_attributes(self):
        # Is there a way to do this without loading the data? Or do only when training begins?
        # Load data
        samples = np.load(self.data_file)
        
        # Total sample size
        self.num_samples = len(samples)
        
        # Scaling factor
        samples = torch.unsqueeze(torch.tensor(samples), 1).float()
        
        if self.transforms is not None:
            for transform in self.transforms:
                samples = transform(samples)
        
        self.scaling_factor = 1/max([torch.max(samples), torch.abs(torch.min(samples))])
    
    def setup(self, stage=None):
        # Load data
        samples = np.load(self.data_file)
        
        if self.truncate_ratio is not None:
            samples = samples[:int(self.truncate_ratio*len(samples))]
        
        self.samples = torch.unsqueeze(torch.tensor(samples), 1).float()
        
        if self.transforms is not None:
            for transform in self.transforms:
                self.samples = transform(self.samples)
        
        self.scale_samples()
        
        # Assign train/val datasets
        if stage == "fit" or stage is None:
            self.inputData_train, self.inputData_val = random_split(self.samples, [self.train_split, 1-self.train_split])

        # Assign test dataset
        if stage == "test" or stage is None:
            self.inputData_test = self.samples
           
    def scale_samples(self):
        self.samples = self.samples*self.scaling_factor
        
    def len(self):
        # Returns length of data (number of samples)
        return self.num_samples

    def num_training_batches(self):
        # Returns number of training batches
        return np.ceil(self.num_samples*self.train_split/self.batch_size)

    def train_dataloader(self):
        # Initiates and returns data loader for training
        return DataLoader(self.inputData_train, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        # Initiates and returns data loader for validation
        return DataLoader(self.inputData_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        # Initiates and returns data loader for testing
        return DataLoader(self.inputData_test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
    

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, data_dir="./data"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

"""Transforms"""
def log10(x):
    return torch.log10(x+1)

transform_dict = {
    'log10': log10
}

