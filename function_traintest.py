"""Pytorch Train and Test"""

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





def train(model, device, train_loader, optimizer, model_name):
    model.train()
    loss_list = [] #  creating list to hold loss per batch
    correct_predictions = 0
    total_images = 0
    #  iterating through batches
    for (images, labels) in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        correct_predictions += (torch.max(predictions, 1)[1] == labels).sum().item()
        total_images += len(images)
        #computing loss/how wrong our classifications are
        if model_name == "lenet5" or "alexnet":
            loss = torch.nn.CrossEntropyLoss()(predictions, labels)
        elif model_name == "snnlenet5":
            loss = torch.nn.functional.nll_loss(predictions, labels)
        else:
            print("Model name error")

        loss.backward() #computing gradients, backward through the graph
        optimizer.step() #optimizing weights
        loss_list.append(loss.item()) #to memorize all the losses value

    mean_loss = np.mean(loss_list)
    accuracy = round(correct_predictions/total_images*100, 2)
    return mean_loss, accuracy


def test(model, device, test_loader, model_name):
    model.eval() #testing mode state
    test_loss = 0
    loss_list = []
    correct_predictions = 0
    total_images = 0
    with torch.no_grad():  # preventing gradient calculations since we will not be optimizing, we are in testing mode
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            correct_predictions += (torch.max(predictions, 1)[1] == labels).sum().item()
            total_images += len(images)

            if model_name == "lenet5" or "alexnet":
                loss = torch.nn.CrossEntropyLoss()(predictions, labels)
            elif model_name == "snnlenet5":
                loss = torch.nn.functional.nll_loss(predictions, labels)
            else:
                print("Model name error")
            loss_list.append(loss.item())

    mean_loss = np.mean(loss_list)
    accuracy = round(correct_predictions / total_images * 100, 2)
    return mean_loss, accuracy