"""Homomorphic Encryption"""

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
import function_global
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
from function_model import model_selection
from random import randint
from PIL import Image
from matplotlib.pyplot import imshow
from typing import Dict
from collections.abc import Sequence
from typing import Tuple, List, Optional
from norse.torch.functional import lif_step, lif_feed_forward_step, lif_current_encoder, LIFParameters
from norse.torch import ConstantCurrentLIFEncoder
from norse.torch import LIFParameters, LIFState
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










def encryption_selection(model, model_name="lenet5" , m=1024, p=100, logpath="."):

    def encode_matrix(HE, matrix):
        try: return np.array(list(map(HE.encodeFrac, matrix)))
        except TypeError: return np.array([encode_matrix(HE, m) for m in matrix])
    def decode_matrix(HE, matrix):
        try: return np.array(list(map(HE.decodeFrac, matrix)))
        except TypeError: return np.array([decode_matrix(HE, m) for m in matrix])
    def encrypt_matrix(HE, matrix):
        try: return np.array(list(map(HE.encryptFrac, matrix)))
        except TypeError: return np.array([encrypt_matrix(HE, m) for m in matrix])
    def decrypt_matrix(HE, matrix):
        try: return np.array(list(map(HE.decryptFrac, matrix)))
        except TypeError: return np.array([decrypt_matrix(HE, m) for m in matrix])

    class HE:
        def generate_keys(self): pass
        def generate_relin_keys(self): pass
        def get_public_key(self): pass
        def get_relin_key(self): pass
        def load_public_key(self, key): pass
        def load_relin_key(self, key): pass
        def encode_matrix(self, matrix): pass
        def decode_matrix(self, matrix): pass
        def encrypt_matrix(self, matrix): pass
        def decrypt_matrix(self, matrix): pass
        def encode_number(self, number): pass
        def power(self, number, exp): pass
        def noise_budget(self, ciphertext): pass

    class BFVPyfhel(HE): #Brakerski/Fan-Vercauteren scheme
        def __init__(self, m, p, sec=128, int_digits=32, frac_digits=16):
            self.he = Pyfhel()
            self.he.contextGen(p, m=m, sec=sec, fracDigits=frac_digits, intDigits=int_digits)
        def generate_keys(self):
            self.he.keyGen()
        def generate_relin_keys(self, bitCount=60, size=3):
            self.he.relinKeyGen(bitCount, size)
        def get_public_key(self):
            self.he.savepublicKey(logpath + "/pub.key")
            with open(logpath + "/pub.key", 'rb') as f: return f.read()
        def get_relin_key(self):
            self.he.saverelinKey(logpath + "/relin.key")
            with open(logpath + "/relin.key", 'rb') as f: return f.read()
        def load_public_key(self, key):
            with open(logpath + "/pub.key", 'wb') as f: f.write(key)
            self.he.restorepublicKey(logpath + "/pub.key")
        def load_relin_key(self, key):
            with open(logpath + "/relin.key", 'wb') as f: f.write(key)
            self.he.restorerelinKey(logpath + "/relin.key")
        def encode_matrix(self, matrix):
            try: return np.array(list(map(self.he.encodeFrac, matrix)))
            except TypeError: return np.array([self.encode_matrix(m) for m in matrix])
        def decode_matrix(self, matrix):
            try: return np.array(list(map(self.he.decodeFrac, matrix)))
            except TypeError: return np.array([self.decode_matrix(m) for m in matrix])
        def encrypt_matrix(self, matrix):
            try: return np.array(list(map(self.he.encryptFrac, matrix)))
            except TypeError: return np.array([self.encrypt_matrix(m) for m in matrix])
        def decrypt_matrix(self, matrix):
            try: return np.array(list(map(self.he.decryptFrac, matrix)))
            except TypeError: return np.array([self.decrypt_matrix(m) for m in matrix])
        def encode_number(self, number): return self.he.encode(number)
        def power(self, number, exp): return self.he.power(number, exp)
        def noise_budget(self, ciphertext):
            try: return self.he.noiseLevel(ciphertext)
            except SystemError: return "Can't get NB without secret key."

    HE_Client = BFVPyfhel(m=m, p=p)
    HE_Client.generate_keys()
    HE_Client.generate_relin_keys()
    public_key = HE_Client.get_public_key()
    relin_key = HE_Client.get_relin_key()
    HE_Server = BFVPyfhel(m=m, p=p)
    HE_Server.load_public_key(public_key)
    HE_Server.load_relin_key(relin_key)

    def apply_padding(t, padding):
        y_p = padding[0]
        x_p = padding[1]
        zero = t[0][0][y_p + 1][x_p + 1] - t[0][0][y_p + 1][x_p + 1]
        return [[np.pad(mat, ((y_p, y_p), (x_p, x_p)), 'constant', constant_values=zero) for mat in layer] for layer in t]
    def convolute2d(image, filter_matrix, stride):
        x_d = len(image[0])
        y_d = len(image)
        x_f = len(filter_matrix[0])
        y_f = len(filter_matrix)
        y_stride = stride[0]
        x_stride = stride[1]
        x_o = ((x_d - x_f) // x_stride) + 1
        y_o = ((y_d - y_f) // y_stride) + 1
        def get_submatrix(matrix, x, y):
            index_row = y * y_stride
            index_column = x * x_stride
            return matrix[index_row: index_row + y_f, index_column: index_column + x_f]
        return np.array([[np.sum(get_submatrix(image, x, y) * filter_matrix) for x in range(0, x_o)] for y in range(0, y_o)])
    def _avg(HE, image, kernel_size, stride):
        x_s = stride[1]
        y_s = stride[0]
        x_k = kernel_size[1]
        y_k = kernel_size[0]
        x_d = len(image[0])
        y_d = len(image)
        x_o = ((x_d - x_k) // x_s) + 1
        y_o = ((y_d - y_k) // y_s) + 1
        denominator = HE.encode_number(1 / (x_k * y_k))
        def get_submatrix(matrix, x, y):
            index_row = y * y_s
            index_column = x * x_s
            return matrix[index_row: index_row + y_k, index_column: index_column + x_k]
        return [[np.sum(get_submatrix(image, x, y)) * denominator for x in range(0, x_o)] for y in range(0, y_o)]

    class ConvolutionalLayer:
        def __init__(self, HE, weights, stride=(1, 1), padding=(0, 0), bias=None):
            self.HE = HE
            self.weights = HE.encode_matrix(weights)
            self.stride = stride
            self.padding = padding
            self.bias = bias
            if bias is not None:
                self.bias = HE.encode_matrix(bias)
        def __call__(self, t):
            t = apply_padding(t, self.padding)
            result = np.array([[np.sum([convolute2d(image_layer, filter_layer, self.stride) for image_layer, filter_layer in zip(image, _filter)], axis=0) for _filter in self.weights] for image in t])
            if self.bias is not None:
                return np.array([[layer + bias for layer, bias in zip(image, self.bias)] for image in result])
            else:
                return result
    class AveragePoolLayer:
        def __init__(self, HE, kernel_size, stride=(1, 1), padding=(0, 0)):
            self.HE = HE
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
        def __call__(self, t):
            t = apply_padding(t, self.padding)
            return np.array([[_avg(self.HE, layer, self.kernel_size, self.stride) for layer in image] for image in t])
    class FlattenLayer:
        def __call__(self, image):
            dimension = image.shape
            return image.reshape(dimension[0], dimension[1] * dimension[2] * dimension[3])
    class LinearLayer:
        def __init__(self, HE, weights, bias=None):
            self.HE = HE
            self.weights = HE.encode_matrix(weights)
            self.bias = bias
            if bias is not None:
                self.bias = HE.encode_matrix(bias)
        def __call__(self, t):
            result = np.array([[np.sum(image * row) for row in self.weights] for image in t])
            if self.bias is not None:
                result = np.array([row + self.bias for row in result])
            return result

    def conv_layer(layer):
        if layer.bias is None: bias = None
        else: bias = layer.bias.detach().numpy()
        return ConvolutionalLayer(HE_Server, weights=layer.weight.detach().numpy(), stride=layer.stride, padding=layer.padding, bias=bias)
    def lin_layer(layer):
        if layer.bias is None: bias = None
        else: bias = layer.bias.detach().numpy()
        return LinearLayer(HE_Server, layer.weight.detach().numpy(), bias)
    def avg_pool_layer(layer):
        # This proxy is required because in PyTorch an AvgPool2d can have kernel_size, stride and padding either of
        # type (int, int) or int, unlike in Conv2d
        kernel_size = (layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size, int) else layer.kernel_size
        stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride
        padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding
        return AveragePoolLayer(HE_Server, kernel_size, stride, padding)
    def flatten_layer(layer):
        return FlattenLayer()

    # equal to lenet5 but each linear layer is encrypted - relu layer is not linear so we need to decrypt again
    class HE_NET_lenet5:
        def __init__(self, HE_Server, HE_Client, model): #uses the normal weights
            self.HE_Server = HE_Server
            self.HE_Client = HE_Client
            self.model = model

        def __call__(self, x, d, debug=False): #x is the encrypted image, d is the plaintext image
            function_global.init()
            function_global.noise_budget.append([])
            function_global.noise_budget[0].append(self.HE_Client.noise_budget(x.item(0)))

            # -------------
            # INPUT - Convolution 1
            # -------------
            x = conv_layer(self.model.conv1)(x)
            d = self.model.conv1(d)
            function_global.noise_budget[0].append(self.HE_Client.noise_budget(x.item(0)))
            x = HE_Client.decrypt_matrix(x)
            x = torch.from_numpy(x)
            function_global.conv1_err.append((x - d).abs().mean().item())
            # Activation
            x = self.model.act1(x)
            d = self.model.act1(d)
            function_global.act1_err.append((x - d).abs().mean().item())

            # -------------
            # LAYER 1 - Pooling 1
            # -------------
            x = HE_Client.encrypt_matrix(x.detach().numpy())
            function_global.noise_budget[0].append(self.HE_Client.noise_budget(x.item(0)))
            x = avg_pool_layer(self.model.pool1)(x)
            d = self.model.pool1(d)
            function_global.noise_budget[0].append(self.HE_Client.noise_budget(x.item(0)))
            x1 = HE_Client.decrypt_matrix(x)
            x1 = torch.from_numpy(x1)
            function_global.pool1_err.append((x1 - d).abs().mean().item())

            # -------------
            # LAYER 2 - Convolution 2
            # -------------
            x = conv_layer(self.model.conv2)(x)
            d = self.model.conv2(d)
            function_global.noise_budget[0].append(self.HE_Client.noise_budget(x.item(0)))
            x = HE_Client.decrypt_matrix(x)
            x = torch.from_numpy(x)
            function_global.conv2_err.append((x - d).abs().mean().item())
            # Activation
            x = self.model.act2(x)
            d = self.model.act2(d)
            function_global.act2_err.append((x - d).abs().mean().item())

            # -------------
            # LAYER 3 - Pooling 2
            # -------------
            x = HE_Client.encrypt_matrix(x.detach().numpy())
            function_global.noise_budget[0].append(self.HE_Client.noise_budget(x.item(0)))
            x = avg_pool_layer(self.model.pool2)(x)
            d = self.model.pool2(d)
            function_global.noise_budget[0].append(self.HE_Client.noise_budget(x.item(0)))
            x1 = HE_Client.decrypt_matrix(x)
            x1 = torch.from_numpy(x1)
            function_global.pool2_err.append((x1 - d).abs().mean().item())

            # -------------
            # LAYER 4 - Flatten
            # -------------
            x = flatten_layer(self.model.flat)(x)
            d = self.model.flat(d)
            function_global.noise_budget[0].append(self.HE_Client.noise_budget(x.item(0)))
            x1 = HE_Client.decrypt_matrix(x)
            x1 = torch.from_numpy(x1)
            function_global.flat_err.append((x1 - d).abs().mean().item())

            # -------------
            # LAYER 5 - Linear 1
            # -------------
            x = lin_layer(self.model.lin1)(x)
            d = self.model.lin1(d)
            function_global.noise_budget[0].append(self.HE_Client.noise_budget(x.item(0)))
            x = HE_Client.decrypt_matrix(x)
            x = torch.from_numpy(x)
            function_global.lin1_err.append((x - d).abs().mean().item())
            # Activation
            x = self.model.act3(x)
            d = self.model.act3(d)
            function_global.act3_err.append((x - d).abs().mean().item())

            # -------------
            # LAYER 6 - Linear 2
            # -------------
            x = HE_Client.encrypt_matrix(x.detach().numpy())
            function_global.noise_budget[0].append(self.HE_Client.noise_budget(x.item(0)))
            x = lin_layer(self.model.lin2)(x)
            d = self.model.lin2(d)
            function_global.noise_budget[0].append(self.HE_Client.noise_budget(x.item(0)))
            x = HE_Client.decrypt_matrix(x)
            x = torch.from_numpy(x)
            function_global.lin2_err.append((x - d).abs().mean().item())
            # Activation
            x = self.model.act4(x)
            d = self.model.act4(d)
            function_global.act4_err.append((x - d).abs().mean().item())

            # -------------
            # LAYER 7 - Linear 3
            # -------------
            x = self.model.lin3(x.float())
            d = self.model.lin3(d)
            function_global.lin3_err.append((x - d).abs().mean().item())

            x = torch.nn.functional.log_softmax(x, dim=1)
            return x

    # equal to normal alexnet
    class HE_NET_alexnet:
        def __init__(self, HE_Server, HE_Client, model):
            self.HE_Server = HE_Server
            self.HE_Client = HE_Client
            self.model = model

        def __call__(self, x, d, debug=False):
            function_global.init()
            function_global.noise_budget.append([])
            function_global.noise_budget[0].append(self.HE_Client.noise_budget(x.item(0)))

            # -------------
            # INPUT -
            # -------------
            x = conv_layer(self.model.conv1)(x)
            d = self.model.conv1(d)
            x = HE_Client.decrypt_matrix(x)
            x = torch.from_numpy(x)
            # Normalization
            x = self.model.norm1(x)
            d = self.model.norm1(d)
            # Activation
            x = self.model.act1(x)
            d = self.model.act1(d)
            x = self.model.pool1(x)
            d = self.model.pool1(d)

            # -------------
            # LAYER 2 -
            # -------------
            x = HE_Client.encrypt_matrix(x.detach().numpy())
            x = conv_layer(self.model.conv2)(x)
            d = self.model.conv2(d)
            x = HE_Client.decrypt_matrix(x)
            x = torch.from_numpy(x)
            # Normalization
            x = self.model.norm2(x)
            d = self.model.norm2(d)
            # Activation
            x = self.model.act2(x)
            d = self.model.act2(d)
            # Pooling
            x = self.model.pool2(x)
            d = self.model.pool2(d)

            # -------------
            # LAYER 3 -
            # -------------
            x = HE_Client.encrypt_matrix(x.detach().numpy())
            x = conv_layer(self.model.conv3)(x)
            d = self.model.conv3(d)
            x = HE_Client.decrypt_matrix(x)
            x = torch.from_numpy(x)
            # Normalization
            x = self.model.norm3(x)
            d = self.model.norm3(d)
            # Activation
            x = self.model.act3(x)
            d = self.model.act3(d)

            # -------------
            # LAYER 4 -
            # -------------
            x = HE_Client.encrypt_matrix(x.detach().numpy())
            x = conv_layer(self.model.conv4)(x)
            d = self.model.conv4(d)
            x = HE_Client.decrypt_matrix(x)
            x = torch.from_numpy(x)
            # Normalization
            x = self.model.norm4(x)
            d = self.model.norm4(d)
            # Activation
            x = self.model.act4(x)
            d = self.model.act4(d)

            # -------------
            # LAYER 5 -
            # -------------
            x = HE_Client.encrypt_matrix(x.detach().numpy())
            x = conv_layer(self.model.conv5)(x)
            d = self.model.conv5(d)
            x = HE_Client.decrypt_matrix(x)
            x = torch.from_numpy(x)
            # Normalization
            x =self.model.norm5(x)
            d = self.model.norm5(d)
            # Activation
            x = self.model.act5(x)
            d = self.model.act5(d)
            # Pooling
            x = self.model.pool5(x)
            d = self.model.pool5(d)

            # -------------
            # LAYER 6 -
            # -------------
            # Dropout
            x = self.model.drop6(x)
            d = self.model.drop6(d)
            # Linear
            x = HE_Client.encrypt_matrix(x.detach().numpy())
            x = lin_layer(self.model.lin6)(x)
            d = self.model.lin6(d)
            x = HE_Client.decrypt_matrix(x)
            x = torch.from_numpy(x)
            # Activation
            x = self.model.act6(x)
            d = self.model.act6(d)

            # -------------
            # LAYER 7 -
            # -------------
            # Dropout
            x = self.model.drop8(x)
            d = self.model.drop8(d)
            # Linear
            x = HE_Client.encrypt_matrix(x.detach().numpy())
            x = lin_layer(self.model.lin8)(x)
            d = self.model.lin8(d)
            x = HE_Client.decrypt_matrix(x)
            x = torch.from_numpy(x)
            # Activation
            x = self.model.act8(x)
            d = self.model.act8(d)

            # -------------
            # LAYER 8 -
            # -------------
            # Linear
            x = self.model.lin8(x.float())
            d = self.model.lin8(d)

            x = torch.nn.functional.log_softmax(x, dim=1)
            return x

    #spiking version
    class HE_NET_snnlenet5:
        def __init__(self, HE_Server, HE_Client, model):
            self.HE_Server = HE_Server
            self.HE_Client = HE_Client
            self.model = model

        def __call__(self, xt, dt, debug=False):
            s0 = s1 = s2 = s3 = so = None
            d0 = d1 = d2 = d3 = do = None
            temp_length = xt.shape[0]
            enc_volt = torch.zeros(temp_length, 1, 10)
            function_global.init()

            for temp in range(temp_length):
                x = xt[temp, :]
                d = dt[temp, :]
                function_global.noise_budget.append([])
                function_global.noise_budget[temp].append(self.HE_Client.noise_budget(x.item(0)))

                # -------------
                # INPUT - Convolution 1
                # -------------
                x = conv_layer(self.model.conv1)(x)
                d = self.model.conv1(d)
                function_global.noise_budget[temp].append(self.HE_Client.noise_budget(x.item(0)))
                x = HE_Client.decrypt_matrix(x)
                x = torch.from_numpy(x)
                function_global.conv1_err.append((x-d).abs().mean().item())
                # Activation
                x, s0 = self.model.lif1(x, s0)
                d, d0 = self.model.lif1(d, d0)
                function_global.act1_err.append((x - d).abs().mean().item())

                # -------------
                # LAYER 1 - Pooling 1
                # -------------
                x = HE_Client.encrypt_matrix(x.detach().numpy())
                function_global.noise_budget[temp].append(self.HE_Client.noise_budget(x.item(0)))
                x = avg_pool_layer(self.model.pool1)(x)
                d = self.model.pool1(d)
                function_global.noise_budget[temp].append(self.HE_Client.noise_budget(x.item(0)))
                x1 = HE_Client.decrypt_matrix(x)
                x1 = torch.from_numpy(x1)
                function_global.pool1_err.append((x1 - d).abs().mean().item())

                # -------------
                # LAYER 2 - Convolution 2
                # -------------
                x = conv_layer(self.model.conv2)(x) * 20
                d = self.model.conv2(d) * 20
                function_global.noise_budget[temp].append(self.HE_Client.noise_budget(x.item(0)))
                x = HE_Client.decrypt_matrix(x)
                x = torch.from_numpy(x)
                function_global.conv2_err.append((x - d).abs().mean().item())
                # Activation
                x, s1 = self.model.lif2(x, s1)
                d, d1 = self.model.lif2(d, d1)
                function_global.act2_err.append((x - d).abs().mean().item())

                # -------------
                # LAYER 3 - Pooling 2
                # -------------
                x = HE_Client.encrypt_matrix(x.detach().numpy())
                function_global.noise_budget[temp].append(self.HE_Client.noise_budget(x.item(0)))
                x = avg_pool_layer(self.model.pool2)(x)
                d = self.model.pool2(d)
                function_global.noise_budget[temp].append(self.HE_Client.noise_budget(x.item(0)))
                x1 = HE_Client.decrypt_matrix(x)
                x1 = torch.from_numpy(x1)
                function_global.pool2_err.append((x1 - d).abs().mean().item())

                # -------------
                # LAYER 4 - Flatten
                # -------------
                x = flatten_layer(self.model.flat)(x)
                d = self.model.flat(d)
                function_global.noise_budget[temp].append(self.HE_Client.noise_budget(x.item(0)))
                x1 = HE_Client.decrypt_matrix(x)
                x1 = torch.from_numpy(x1)
                function_global.flat_err.append((x1 - d).abs().mean().item())

                # -------------
                # LAYER 5 - Linear 1
                # -------------
                x = lin_layer(self.model.lin1)(x)
                d = self.model.lin1(d)
                function_global.noise_budget[temp].append(self.HE_Client.noise_budget(x.item(0)))
                x = HE_Client.decrypt_matrix(x)
                x = torch.from_numpy(x)
                function_global.lin1_err.append((x - d).abs().mean().item())
                # Activation
                x, s2 = self.model.lif3(x, s2)
                d, d2 = self.model.lif3(d, d2)
                function_global.act3_err.append((x - d).abs().mean().item())

                # -------------
                # LAYER 6 - Linear 2
                # -------------
                x = HE_Client.encrypt_matrix(x.detach().numpy())
                function_global.noise_budget[temp].append(self.HE_Client.noise_budget(x.item(0)))
                x = lin_layer(self.model.lin2)(x)
                d = self.model.lin2(d)
                function_global.noise_budget[temp].append(self.HE_Client.noise_budget(x.item(0)))
                x = HE_Client.decrypt_matrix(x)
                x = torch.from_numpy(x)
                function_global.lin2_err.append((x - d).abs().mean().item())
                # Activation
                x, s3 = self.model.lif4(x, s3)
                d, d3 = self.model.lif4(d, d3)
                function_global.act4_err.append((x - d).abs().mean().item())

                # -------------
                # LAYER 7 - Linear 3
                # -------------
                x, so = self.model.lin3(torch.nn.functional.relu(x.float()), so)
                d, do = self.model.lin3(torch.nn.functional.relu(d), do)
                function_global.lin3_err.append((x - d).abs().mean().item())

                enc_volt[temp, :, :] = x
            return enc_volt


    if model_name == "lenet5":
        return HE_NET_lenet5(HE_Server, HE_Client, model), HE_Client
    elif model_name == "alexnet":
        return HE_NET_alexnet(HE_Server, HE_Client, model), HE_Client
    elif model_name == "snnlenet5":
        return HE_NET_snnlenet5(HE_Server, HE_Client, model.snn), HE_Client
    else:
        print("Error: choose a correct model between: lenet5 - snnlenet5")