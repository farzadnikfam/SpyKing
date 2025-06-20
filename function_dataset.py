"""Dataset download and normalization"""

import datetime
import torch
import torchvision
import json
import cv2
import time
import pathlib
import PIL
import random
import os
import sys
import gzip
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from tqdm.notebook import tqdm, trange
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
from Pyfhel.util import ENCODING_t
from pathlib import Path










# datasets normalization
def dataset_selection(dataset_name, dataset_path, model_name = "lenet5", batch_size=256):
    if dataset_name == "cifar10":
        if model_name == "alexnet" or model_name == "snnalexnet":
            transform = transforms.Compose([torchvision.transforms.ToTensor(), transforms.Resize((227,227)), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            transform = transforms.Compose([torchvision.transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)
        n_classes = len(train.classes)

    elif dataset_name == "mnist":
        if model_name == "alexnet" or model_name == "snnalexnet":
            transform = transforms.Compose([torchvision.transforms.ToTensor(), transforms.Resize((227,227)), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            transform = transforms.Compose([torchvision.transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
        test = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True, transform=transform)
        n_classes = len(train.classes)

    elif dataset_name == "fashion":
        if model_name == "alexnet" or model_name == "snnalexnet":
            transform = transforms.Compose([torchvision.transforms.ToTensor(), transforms.Resize((227, 227)), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            transform = transforms.Compose([torchvision.transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = torchvision.datasets.FashionMNIST(root=dataset_path, train=True, download=True, transform=transform)
        test = torchvision.datasets.FashionMNIST(root=dataset_path, train=False, download=True, transform=transform)
        n_classes = len(train.classes)

    else: print("dataset error, choose between: cifar10 - mnist - fashion")

    return train, test, n_classes