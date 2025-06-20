"""Pytorch Models"""

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
import tempfile
import multiprocessing
import pickle
import math
import numbers
import random
import warnings
import contextlib
import jsonpickle
import json
import matplotlib.pyplot as plt
import platform as plat
import pandas as pd

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
from function_telegram_bot import telegram_bot_text as bottex
from function_telegram_bot import telegram_bot_image as botima
from function_telegram_bot import telegram_bot_file as botfil
from function_dataset import dataset_selection
from random import randint
from PIL import Image
from matplotlib.pyplot import imshow
from typing import Dict
from collections.abc import Sequence
from typing import Tuple, List, Optional
from norse.torch.functional import lif_step, lif_feed_forward_step, lif_current_encoder, LIFParameters
from norse.torch import ConstantCurrentLIFEncoder
from norse.torch import LIFState, LIFParameters
from norse.torch.module.lif import LIFCell, LIF
from norse.torch import LICell, LIState
from norse.torch.module import SequentialState
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.functional.lif import LIFFeedForwardState
from norse.torch.functional.leaky_integrator import LIState
from typing import NamedTuple
from norse.torch import SpikeLatencyLIFEncoder
from norse.torch import PoissonEncoder

import torch.nn as nn
import torch.nn.functional as F










#LeNet5 model
class LeNet5(nn.ModuleList):
    #the model scheme is present in the paper

    def __init__(self, dataset_name = "mnist"):
        super(LeNet5, self).__init__()
        if dataset_name == "cifar10":
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        elif dataset_name == "mnist" or dataset_name == "fashion":
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        else:
            print("Dataset error")
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.flat = nn.Flatten()
        if dataset_name == "cifar10":
            self.lin1 = nn.Linear(16 * 5 * 5, 120)
        elif dataset_name == "mnist" or dataset_name == "fashion":
            self.lin1 = nn.Linear(16 * 4 * 4, 120)
        else:
            print("Dataset error")
        self.act3 = nn.ReLU()
        self.lin2 = nn.Linear(in_features=120, out_features=84)
        self.act4 = nn.ReLU()
        self.lin3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # -------------
        # INPUT - Convolution 1
        # -------------
        x = self.conv1(x)
        x = self.act1(x)
        # -------------
        # LAYER 1 - Pooling 1
        # -------------
        x = self.pool1(x)
        # -------------
        # LAYER 2 - Convolution 2
        # -------------
        x = self.conv2(x)
        x = self.act2(x)
        # -------------
        # LAYER 3 - Pooling 2
        # -------------
        x = self.pool2(x)
        # -------------
        # LAYER 4 - Flatten
        # -------------
        x = self.flat(x)
        # -------------
        # LAYER 5 - Linear 1
        # -------------
        x = self.lin1(x)
        x = self.act3(x)
        # -------------
        # LAYER 6 - Linear 2
        # -------------
        x = self.lin2(x)
        x = self.act4(x)
        # -------------
        # LAYER 7 - Linear 3
        # -------------
        x = self.lin3(x)
        # --------------
        # OUTPUT LAYER
        # --------------
        return x

"""
#with fashion mnist
LeNet5(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (act1): ReLU()
  (pool1): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (act2): ReLU()
  (pool2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (flat): Flatten(start_dim=1, end_dim=-1)
  (lin): Linear(in_features=256, out_features=120, bias=True)
  (act3): ReLU()
  (lin1): Linear(in_features=120, out_features=84, bias=True)
  (act4): ReLU()
  (lin2): Linear(in_features=84, out_features=10, bias=True)
)
"""

class AlexNet(nn.Module):
    def __init__(self, dataset_name = "mnist", num_classes=10):
        super(AlexNet, self).__init__()

        if dataset_name == "cifar10":
            input_layer = 3
        elif dataset_name == "mnist" or dataset_name == "fashion":
            input_layer = 1
        else:
            print("Dataset error")

        ### layer 1
        self.conv1 = nn.Conv2d(input_layer, 96, kernel_size=11, stride=4, padding=0)
        self.norm1 = nn.BatchNorm2d(96)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        ### layer 2
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.BatchNorm2d(256)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        ### layer 3
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(384)
        self.act3 = nn.ReLU()

        ### layer 4
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(384)
        self.act4 = nn.ReLU()

        ### layer 5
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(256)
        self.act5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        ### layer 6
        self.drop6 = nn.Dropout(0.5)
        self.lin6 = nn.Linear(9216, 4096)
        self.act6 = nn.ReLU()

        ### layer 7
        self.drop7 = nn.Dropout(0.5)
        self.lin7 = nn.Linear(4096, 4096)
        self.act7 = nn.ReLU()

        ### layer 8
        self.lin8 = nn.Linear(4096, num_classes)


    def forward(self, x): #torch.Size([10, 1, 227, 227])
        x = self.conv1(x) #torch.Size([10, 96, 55, 55])
        x = self.norm1(x) #torch.Size([10, 96, 55, 55])
        x = self.act1(x) #torch.Size([10, 96, 55, 55])
        x = self.pool1(x) #torch.Size([10, 96, 27, 27])

        x = self.conv2(x) #torch.Size([10, 256, 27, 27])
        x = self.norm2(x) #torch.Size([10, 256, 27, 27])
        x = self.act2(x) #torch.Size([10, 256, 27, 27])
        x = self.pool2(x) #torch.Size([10, 256, 13, 13])

        x = self.conv3(x) #torch.Size([10, 384, 13, 13])
        x = self.norm3(x) #torch.Size([10, 384, 13, 13])
        x = self.act3(x) #torch.Size([10, 384, 13, 13])

        x = self.conv4(x) #torch.Size([10, 384, 13, 13])
        x = self.norm4(x) #torch.Size([10, 384, 13, 13])
        x = self.act4(x) #torch.Size([10, 384, 13, 13])

        x = self.conv5(x) #torch.Size([10, 256, 13, 13])
        x = self.norm5(x) #torch.Size([10, 256, 13, 13])
        x = self.act5(x) #torch.Size([10, 256, 13, 13])
        x = self.pool5(x) #torch.Size([10, 256, 6, 6])

        x = x.reshape(x.size(0), -1) #torch.Size([10, 9216])

        x = self.drop6(x) #torch.Size([10, 9216])
        x = self.lin6(x) #torch.Size([10, 4096])
        x = self.act6(x) #torch.Size([10, 4096])

        x = self.drop7(x) #torch.Size([10, 4096])
        x = self.lin7(x) #torch.Size([10, 4096])
        x = self.act7(x) #torch.Size([10, 4096])

        x = self.lin8(x) #torch.Size([10, 10])

        return x


class SNN_LeNet5(nn.ModuleList):

    def __init__(self, lifparameters, dataset_name = "mnist"):
        super(SNN_LeNet5, self).__init__()
        if dataset_name == "cifar10":
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        elif dataset_name == "mnist" or dataset_name == "fashion":
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        else:
            print("Dataset error")
        self.lif1 = LIFCell(lifparameters)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.lif2 = LIFCell(lifparameters)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.flat = nn.Flatten()
        if dataset_name == "cifar10":
            self.lin1 = nn.Linear(16 * 5 * 5, 120)
        elif dataset_name == "mnist" or dataset_name == "fashion":
            self.lin1 = nn.Linear(16 * 4 * 4, 120)
        else:
            print("Dataset error")
        self.lif3 = LIFCell(lifparameters)
        self.lin2 = nn.Linear(in_features=120, out_features=84)
        self.lif4 = LIFCell(lifparameters)
        self.lin3 = LILinearCell(84, 10)

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0 = s1 = s2 = s3 = so = None

        voltages = torch.zeros(seq_length, batch_size, 10, device=x.device, dtype=x.dtype)

        for ts in range(seq_length):
            z = self.conv1(x[ts, :])
            z, s0 = self.lif1(z, s0)
            z = self.pool1(z)
            z = 20 * self.conv2(z)
            z, s1 = self.lif2(z, s1)
            z = self.pool2(z)
            z = self.flat(z)
            z = self.lin1(z)
            z, s2 = self.lif3(z, s2)
            z = self.lin2(z)
            z, s3 = self.lif4(z, s3)
            v, so = self.lin3(torch.nn.functional.relu(z), so)

            voltages[ts, :, :] = v
        return voltages


class SNN_AlexNet(nn.Module):
    def __init__(self, lifparameters, dataset_name="mnist", num_classes=10):
        super(SNN_AlexNet, self).__init__()

        if dataset_name == "cifar10":
            input_layer = 3
        elif dataset_name == "mnist" or dataset_name == "fashion":
            input_layer = 1
        else:
            print("Dataset error")

        ### layer 1
        self.conv1 = nn.Conv2d(input_layer, 96, kernel_size=11, stride=4, padding=0)
        self.norm1 = nn.BatchNorm2d(96)
        self.lif1 = LIFCell(lifparameters)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        ### layer 2
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.norm2 = nn.BatchNorm2d(256)
        self.lif2 = LIFCell(lifparameters)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        ### layer 3
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(384)
        self.lif3 = LIFCell(lifparameters)

        ### layer 4
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(384)
        self.lif4 = LIFCell(lifparameters)

        ### layer 5
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(256)
        self.lif5 = LIFCell(lifparameters)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        ### layer 6
        self.drop6 = nn.Dropout(0.5)
        self.lin6 = nn.Linear(9216, 4096)
        self.lif6 = LIFCell(lifparameters)

        ### layer 7
        self.drop7 = nn.Dropout(0.5)
        self.lin7 = nn.Linear(4096, 4096)
        self.lif7 = LIFCell(lifparameters)

        ### layer 8
        self.lin8 = LILinearCell(4096, num_classes)

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]
        # specify the initial states
        s1 = s2 = s3 = s4 = s5 = s6 = s7 = so = None
        voltages = torch.zeros(seq_length, batch_size, 10, device=x.device, dtype=x.dtype)

        for ts in range(seq_length):
            
            z = self.conv1(x[ts, :]) #torch.Size([10, 96, 55, 55])
            z = self.norm1(z) #torch.Size([10, 96, 55, 55])
            z, s1 = self.lif1(z, s1) #torch.Size([10, 96, 55, 55])
            z = self.pool1(z) #torch.Size([10, 96, 27, 27])
            
            z = 20 * self.conv2(z) #torch.Size([10, 256, 27, 27])
            z = self.norm2(z) #torch.Size([10, 256, 27, 27])
            z, s2 = self.lif2(z, s2) #torch.Size([10, 256, 27, 27])
            z = self.pool2(z) #torch.Size([10, 256, 13, 13])

            z = self.conv3(z) #torch.Size([10, 384, 13, 13])
            z = self.norm3(z) #torch.Size([10, 384, 13, 13])
            z, s3 = self.lif3(z, s3) #torch.Size([10, 384, 13, 13])

            z = self.conv4(z) #torch.Size([10, 384, 13, 13])
            z = self.norm4(z) #torch.Size([10, 384, 13, 13])
            z, s4 = self.lif4(z, s4) #torch.Size([10, 384, 13, 13])

            z = self.conv5(z) #torch.Size([10, 256, 13, 13])
            z = self.norm5(z) #torch.Size([10, 256, 13, 13])
            z, s5 = self.lif5(z, s5) #torch.Size([10, 256, 13, 13])
            z = self.pool5(z) #torch.Size([10, 256, 6, 6])

            z = z.reshape(z.size(0), -1) #torch.Size([10, 9216])

            z = self.drop6(z) #torch.Size([10, 9216])
            z = self.lin6(z) #torch.Size([10, 4096])
            z, s6 = self.lif6(z, s6) #torch.Size([10, 4096])

            z = self.drop7(z) #torch.Size([10, 4096])
            z = self.lin7(z) #torch.Size([10, 4096])
            z, s7 = self.lif7(z, s7) #torch.Size([10, 4096])

            v, so = self.lin8(torch.nn.functional.relu(z), so) #torch.Size([10, 10])

            voltages[ts, :, :] = v
        return voltages


class Model(nn.ModuleList):
    def __init__(self, encoder, snn, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y


def decode(x): #for spiking
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    o = torch.argmax(log_p_y)
    return log_p_y


def model_selection(model_name = "lenet5", dataset_name = "mnist", snn_param = [200, 100, 0, 1, 0], T = 30): #snn standard norse parameters
    if model_name == "lenet5":
        return LeNet5(dataset_name=dataset_name)
    elif model_name == "alexnet":
        return AlexNet(dataset_name=dataset_name)
    elif model_name == "snnlenet5":
        lifparameters = LIFParameters(  tau_syn_inv=torch.as_tensor(snn_param[0]), #inverse of tau syn
                                        tau_mem_inv=torch.as_tensor(snn_param[1]), #inverse of tau mem
                                        v_leak=torch.as_tensor(snn_param[2]), #leak voltage
                                        v_th=torch.as_tensor(snn_param[3]), #threshold voltage
                                        v_reset=torch.as_tensor(snn_param[4])) #reset voltage
        return Model(encoder=ConstantCurrentLIFEncoder(T), snn=SNN_LeNet5(lifparameters, dataset_name), decoder=decode)
    elif model_name == "snnalexnet":
        lifparameters = LIFParameters(  tau_syn_inv=torch.as_tensor(snn_param[0]), #inverse of tau syn
                                        tau_mem_inv=torch.as_tensor(snn_param[1]), #inverse of tau mem
                                        v_leak=torch.as_tensor(snn_param[2]), #leak voltage
                                        v_th=torch.as_tensor(snn_param[3]), #threshold voltage
                                        v_reset=torch.as_tensor(snn_param[4])) #reset voltage
        return Model(encoder=ConstantCurrentLIFEncoder(T), snn=SNN_AlexNet(lifparameters, dataset_name), decoder=decode)

    else:
        print("Error: choose a correct model between: lenet5 - snnlenet5")