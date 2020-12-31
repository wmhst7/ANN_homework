import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os


class Dataset(object):

    def __init__(self, batch_size, path):
        if not os.path.isdir(path):
            os.mkdir(path)

        self._training_data = datasets.MNIST(
            root=path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
            ])
        )

        self._validation_data = datasets.MNIST(
            root=path,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
            ])
        )

        self._training_loader = DataLoader(
            self._training_data,
            batch_size=batch_size,
            num_workers=2,
            shuffle=True,
            pin_memory=True
        )

        self._validation_loader = DataLoader(
            self._validation_data,
            batch_size=batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=False
        )

    @property
    def training_data(self):
        return self._training_data

    @property
    def validation_data(self):
        return self._validation_data

    @property
    def training_loader(self):
        return self._training_loader

    @property
    def validation_loader(self):
        return self._validation_loader
