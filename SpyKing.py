### main file for the SpyKing project

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
import argparse
import shutil
import function_global
import matplotlib.pyplot as plt
import platform as plat
import pandas as pd

from datetime import datetime
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from tqdm.notebook import tqdm, trange
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
from pathlib import Path
from function_telegram_bot import telegram_bot_text
from function_telegram_bot import telegram_bot_image
from function_telegram_bot import telegram_bot_file
from function_telegram_bot import telegram_bot_text_print as teleprint
from function_encryption import encryption_selection
from function_dataset import dataset_selection
from function_model import model_selection, decode
from function_traintest import train, test
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
from function_log import log_directory
from function_log import log_zip
from function_panda_csv import panda_csv

import torch.nn as nn
import torch.nn.functional as F










function_global.init() #to initialize the global variables
# to initialize the telegram bot and send all the information about the system and the current file to the bot
telegram_bot_text("\n====================\n\n\n\n\n\n\n\n\n\n====================")
teleprint("====================\n====================\n=======START========\n====================\n====================")

file_name=os.path.splitext("%s" %(os.path.basename(__file__)))[0] #current file name without extension
teleprint("%s" %file_name, bold = True) #print current file name without extension
teleprint("System: %s" %(plat.system()) + " - Release: %s" %(plat.release()) + " - Version: %s" %(plat.version())) #to print the system information


#import variables from command line
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="lenet5", choices=["lenet5","alexnet","snnlenet5","snnalexnet"], help="the name of the model to be trained")
parser.add_argument('--debugging_mode', '-d', type=int, default=0, choices=[0,1], help="to activate debuggind mode, reducing number of epochs and images")
parser.add_argument('--training_flag', type=int, default=1, choices=[0,1], help="1 or 0 to do or not the training for the model, if the model doesn't exist the training is compulsory")
parser.add_argument('--testing_flag', type=int, default=1, choices=[0,1], help="1 or 0 to do or not the testing for the model after a complete cycle of training")
parser.add_argument('--validation_flag', type=int, default=1, choices=[0,1], help="1 or 0 to do or not the validation on the test set during the training, per each epoch")
parser.add_argument('--model_flag', type=int, default=1, choices=[0,1], help="1 or 0 to use an existing model or not")
parser.add_argument('--saving_flag', type=int, default=1, choices=[0,1], help="1 or 0 to save or not the model")
parser.add_argument('--model_name_flag', type=int, default=0, choices=[0,1], help="1 or 0 to mantain or not the same model name")
parser.add_argument('--logdir_flag', type=int, default=1, choices=[0,1], help="1 or 0 to use a numerate log directory or a temporary one")
parser.add_argument('--image_show_flag', type=int, default=0, choices=[0,1], help="1 or 0 to show or not the current analized image")
parser.add_argument('--image_save_flag', type=int, default=1, choices=[0,1], help="1 or 0 to save or not the current analized image")
parser.add_argument('--image_csv_flag', type=int, default=1, choices=[0,1], help="1 or 0 to save or not the current analized image data in CSV")
parser.add_argument('--encryption_flag', type=int, default=1, choices=[0,1], help="1 or 0 to do or not the encryption and the valutation on the encrypted data")
parser.add_argument('--batch_size', type=int, default=256, help="number of batch size")
parser.add_argument('--traingroup', type=int, default=-1, help="-1 all the training set, otherwise positive integers")
parser.add_argument('--testgroup', type=int, default=-1, help="-1 all the testing set, otherwise positive integers")
parser.add_argument('--validationgroup', type=int, default=-1, help="-1 all the testing set, otherwise positive integers")
parser.add_argument('--device_flag', type=int, default=1, choices=[0,1], help="1 or 0 to choose GPU or CPU accelerator")

parser.add_argument('--lr_finder_flag', type=int, default=0, choices=[0,1], help="1 or 0 to do or not a train to find the best learning rate with an exponential decay")
parser.add_argument('--lr_start', type=float, default=1e-5, help="starting lr for the lr finder, the normal lr will be set to this value")
parser.add_argument('--lr_end', type=float, default=1, help="ending lr to obtain after the exponential decay")
parser.add_argument('--gamma', type=float, default=-1, help="-1 to use lr_start and lr_end to automatic calculate the gamma, otherwise positive float as reducing factor of lr")

parser.add_argument('--model_path', type=str, default="topmodels", help="path where to save the trained model")
parser.add_argument('--dataset_name', type=str, default="fashion", choices=["mnist","fashion","cifar10"], help="dataset to be used")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate, best around 0.001 and 0.0001")
parser.add_argument('--train_epochs', type=int, default=20, help="how many cycle to do for the training")
parser.add_argument('--validation_epochs', type=int, default=1, help="how many train_epochs to do the validation on test set, if train_epochs is 10 and validation_epochs is 2 the validation takes every 2 epochs and 5 times in total")
parser.add_argument('--model_load_name', type=str, default="NULL", help="model name to be loaded with or without the extension .pt, please use name different from 'NULL")

parser.add_argument('--class_minimum', type=int, default=10, help="1000 to analize all the test set, otherwise specify the minimum number of images to analize per each class")
parser.add_argument('--counter', type=int, default=-1, help="-1 to stop at the class_minimum, otherwise is the number of images analized for encryption")
parser.add_argument('--starting_image', type=int, default=0, help="to start analysing from an image other than 0")

parser.add_argument('--m_list', '-m', nargs="+", type=int, default=[1024, 2048], help="a list of Polynomial modulus degree, which requires to be set as a positive integer power of 2 since it is the degree of the cyclotomic polynomial")
parser.add_argument('--p_list', '-p', nargs="+", type=int, default=[10,20,50,100,200,500,1000,2000,5000], help="a list of Plaintext modulus, set as a positive integer, it is the module of the coefficients of the polynomial ring")
parser.add_argument('--time_steps', '-T', type=int, default=30, help="number of time steps for the spiking version")

args = parser.parse_args()

#the p is t in the paper


model_name = args.model_name
debugging_mode = args.debugging_mode
training_flag = args.training_flag
testing_flag = args.testing_flag
validation_flag = args.validation_flag
model_flag = args.model_flag
saving_flag = args.saving_flag
model_name_flag = args.model_name_flag
logdir_flag = args.logdir_flag
image_show_flag = args.image_show_flag
image_save_flag = args.image_save_flag
image_csv_flag = args.image_csv_flag
encryption_flag = args.encryption_flag
batch_size = args.batch_size
traingroup = args.traingroup
testgroup = args.testgroup
validationgroup = args.validationgroup
lr_finder_flag = args.lr_finder_flag
lr_start = args.lr_start
lr_end = args.lr_end
gamma = args.gamma
model_path = args.model_path
dataset_name = args.dataset_name
lr = args.lr
train_epochs = args.train_epochs
validation_epochs = args.validation_epochs
model_load_name = args.model_load_name
class_minimum = args.class_minimum
counter = args.counter
starting_image = args.starting_image
m_list = args.m_list
p_list = args.p_list
T = args.time_steps
device_flag = args.device_flag


#fast testing mode
if debugging_mode == 1:
    """
    write here the code for a fast testing mode, with a reduced number of epochs and images
    """
    pass


#creating model path
try: os.mkdir(model_path)
except: pass

#loading models
if model_load_name == "NULL":
    model_save_name_free = "%s_" %(model_name) + "%s" %(dataset_name)
    model_save_name_ext = "%s_" %(model_name) + "%s.pt" %(dataset_name)
    model_path_free = model_path + "/%s" %(model_save_name_free)
    model_path_ext = model_path + "/%s" %(model_save_name_ext)
elif (model_load_name != "NULL") and (model_load_name.find(".pt") >= 0):
    model_path_free = model_path + "/%s" %(model_load_name.replace(".pt",""))
    model_path_ext = model_path + "/%s" %(model_load_name)
elif (model_load_name != "NULL") and (model_load_name.find(".pt") < 0):
    model_path_free = model_path + "/%s" % (model_load_name)
    model_path_ext = model_path + "/%s.pt" % (model_load_name)
else: print ("Ammè me pare 'na strunzat'")


dataset_path = "datasets"
N = 1 # number of neurons to consider
snn_param = [200.0, 100.0, 0.0, 0.5, 0.0] #[inverse of tau syn, inverse of tau mem, leak voltage, threshold voltage, reset voltage]
ex_encoder = ConstantCurrentLIFEncoder(T) # encoder for the SNN, it is a constant current encoder
print("time steps: %s" %T)

class_verification = [0,1,2,3,4,5,6,7,8,9]
class_counter = [0,0,0,0,0,0,0,0,0,0]

#printing all the information about the system and the current file to the bot
teleprint("Model: %s" %(model_name) + " - Dataset: %s" %(dataset_name) + " - Epochs: %s" %(train_epochs))

#setting the number of workers for the DataLoader
if plat.system() == "Windows": num_workers = 0
else: num_workers = 2

#setting the log directory
if logdir_flag == 0: log_directory(logdir=-1)
logpath = log_directory(logdir_flag, logname=file_name)
teleprint("Logs saved in: %s" %logpath)

#creating the log directory
logwriter = SummaryWriter(logpath)

#creating the panda dataframe for the log
panda_log = pd.DataFrame(columns=['lr', 'acc_train', 'acc_val', 'acc_test', 'loss_train', 'loss_val', 'loss_test'])

#loading datasets
trainset, testset, n_classes = dataset_selection(dataset_name, dataset_path, model_name)

#setting the number of images to be used for training, testing and validation
if traingroup < 0: traingroup = len(trainset)
else: pass
if testgroup < 0: testgroup = len(testset)
else: pass
if validationgroup < 0: validationgroup = len(testset)
else: pass


#learning rate finder mode
if (lr_finder_flag == 1) and (gamma < 0):
    lr = lr_start
    gamma = 1/math.exp(math.log(lr_start/lr_end)/train_epochs)
else: pass

train_loader = DataLoader(torch.utils.data.Subset(trainset, range(0, traingroup)), batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(torch.utils.data.Subset(testset, range(0, testgroup)), batch_size=batch_size, shuffle=False, num_workers=num_workers)
validation_loader = DataLoader(torch.utils.data.Subset(testset, range(0, validationgroup)), batch_size=batch_size, shuffle=False, num_workers=num_workers)

if device_flag == 1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

try:
    if model_flag == 1:
        model = torch.load(model_path_ext).to(device)
        teleprint("Model found in %s" %model_path_ext)
    else:
        raise Exception
except:
    model = model_selection(model_name, dataset_name, snn_param, T).to(device)
    training_flag = 1 #the training with a new model is compulsory
    teleprint("Model not found, new training will start soon")

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck') #cifar10 --- 50000,10000 --- RGB,32,32
classes = ('zero','one','two','three','four','five','six','seven','eight','nine') #mnist --- 60000,10000 --- mono,28,28
classes = ('T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot') #fashion --- 60000,10000 --- mono,28,28


optimizer = optim.Adam(model.parameters(), lr= lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, verbose=False)


#training phase
validation_counter = 0
if training_flag == 1:
    if (validation_flag == 1):
        teleprint("The model will be trained and evaluated every %s epochs" % validation_epochs)
    else:
        teleprint("The model will be trained, but it will NOT be evaluated")
    if (lr_finder_flag == 1):
        teleprint("This is NOT a normal training, it is only for the learning rate finder")
    else:
        teleprint ("This is a normal training")
    for epoch in range(train_epochs):
        mean_loss_training, accuracy_training = train(model, device, train_loader, optimizer, model_name)
        logwriter.add_scalar('accuracy/train', accuracy_training, epoch)
        logwriter.add_scalar('loss/train', mean_loss_training, epoch)
        logwriter.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        panda_log.loc[epoch,'acc_train'] = accuracy_training
        panda_log.loc[epoch,'loss_train'] = mean_loss_training
        panda_log.loc[epoch,'lr'] = scheduler.get_last_lr()[0]
        validation_counter += 1
        if lr_finder_flag == 1: scheduler.step()
        else: pass
        if (validation_flag == 1) and (validation_counter == validation_epochs):
            mean_loss_validation, accuracy_validation = test(model, device, validation_loader, model_name)
            logwriter.add_scalar('accuracy/validation', accuracy_validation, epoch)
            logwriter.add_scalar('loss/validation', mean_loss_validation, epoch)
            panda_log.loc[epoch, 'acc_val'] = accuracy_validation
            panda_log.loc[epoch, 'loss_val'] = mean_loss_validation
            validation_counter = 0
            teleprint('%05d' %epoch + ' -- lr: %.5f' %scheduler.get_last_lr()[0] + ' - acc: %.2f' %accuracy_training + ' - val: %.2f' %accuracy_validation)
        else:
            teleprint('%05d' %epoch + ' -- lr: %.5f' %scheduler.get_last_lr()[0] + ' - acc: %.2f' %accuracy_training)
else:
    teleprint("The model will NOT be trained")


#testing phase
if testing_flag == 1:
    teleprint("The testing flag is on, the model will be tested")
    mean_loss_testing, accuracy_testing = test(model, device, test_loader, model_name)
    logwriter.add_scalar('accuracy/test', accuracy_testing)
    logwriter.add_scalar('loss/test', mean_loss_testing)
    panda_log['acc_test'] = accuracy_testing
    panda_log['loss_test'] = mean_loss_testing
    teleprint("Testing accuracy: %.2f" % accuracy_testing)
else:
    teleprint("The testing flag is off, the model will NOT be tested")


#model saving
if saving_flag == 1:
    if model_name_flag == 1:
        torch.save(model,model_path_ext)
        teleprint("Model saved in: %s" %model_path_ext)
    else:
        model_path_ext = model_path_free + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pt"
        torch.save(model,model_path_ext)
        teleprint("Model saved in: %s" %model_path_ext)
else:
    teleprint("Model not saved")


#how many images to check with encryption
if encryption_flag == 1:
    if counter == -1:
        for counter in range(len(testset.targets)):
            class_counter[testset.targets[counter + starting_image]] += 1
            stopping_flag = 0
            for k in range(len(class_counter)):
                if class_counter[k]>=class_minimum: stopping_flag += 1
            if stopping_flag == n_classes: break
        counter += 1
    else: pass

    if counter + starting_image > 10000:
        max_counter = 10000
    else:
        max_counter = counter + starting_image
    batch_images, batch_labels = next(iter(DataLoader(torch.utils.data.Subset(testset, range(0, max_counter)), batch_size=max_counter, shuffle=False, num_workers=num_workers)))

    model.cpu()
    model.eval()

else: pass





#encryption part
print_counter = 0
if encryption_flag == 1:
    teleprint("The encrypted evaluation will be done on:\n%s images\n" % counter + "m parameter [%s]\n" %', '.join(str(m) for m in m_list) + "p parameter [%s]" %', '.join(str(p) for p in p_list))
    for m in m_list:
        for p in p_list:
            encr_model, HE_Client = encryption_selection(model, model_name, m, p, logpath)
            teleprint("-------------------------")
            teleprint("-------------------------")

            for number in range(starting_image,len(batch_images)):

                single_image = batch_images[number].cpu()
                single_label = batch_labels[number].cpu().item()

                teleprint("-------------------------")
                teleprint("%s - " %model_name + "%s " % dataset_name + "- image: %05d " % number + "- label: %s" % single_label, bold = True)
                teleprint("m: %d - " %m + "p: %d" %p)


                #image encryption
                time_start = time.time()
                if model_name == "lenet5" or model_name == "alexnet":
                    encr_image = HE_Client.encrypt_matrix(single_image.unsqueeze(0).numpy())
                elif model_name == "snnlenet5" or model_name == "snnalexnet":
                    single_image_temp = ex_encoder(single_image)
                    encr_image = HE_Client.encrypt_matrix(single_image_temp.unsqueeze(1).numpy())
                time_encryption = round(time.time() - time_start, 2)

                if print_counter < len(batch_images):
                    plt.figure(number)
                    plt.axes().set_axis_off()
                    plt.imshow(single_image.permute(1, 2, 0), cmap='gray')
                    if image_save_flag == 1:
                        plt.savefig(logpath + "/dataset_%s__" %dataset_name + "image_%05d__" %number + "label_%s" %single_label, bbox_inches='tight')
                    else: pass
                    if image_show_flag == 1: pass
                    else: plt.close()

                    if model_name == "snnlenet5" or model_name == "snnalexnet":
                        for t in range(T):
                            plt.figure("%s_" %number + "%s" %t)
                            plt.axes().set_axis_off()
                            plt.imshow(single_image_temp[t].permute(1, 2, 0), cmap='gray')
                            if image_save_flag == 1:
                                plt.savefig(logpath + "/dataset_%s__" %dataset_name + "image_%05d__" %number + "label_%s__" %single_label + "temp_%02d" %t, bbox_inches='tight')
                            else: pass
                            if image_show_flag == 1: pass
                            else: plt.close()
                        plt.figure("%s_sum" % number)
                        plt.axes().set_axis_off()
                        plt.imshow(single_image_temp.sum(0).permute(1, 2, 0), cmap='gray')
                        if image_save_flag == 1:
                            plt.savefig(logpath + "/dataset_%s__" %dataset_name + "image_%05d__" %number + "label_%s__" %single_label + "temp_sum", bbox_inches='tight')
                        else: pass
                        if image_show_flag == 1: pass
                        else: plt.close()
                    print_counter += 1


                #plaintext execution
                time_start = time.time()
                with torch.no_grad():
                    if model_name == "lenet5" or model_name == "alexnet":
                        normal_predictions = model(single_image.unsqueeze(0))
                        normal_predictions = torch.nn.functional.log_softmax(normal_predictions, dim=1)
                    elif model_name == "snnlenet5" or model_name == "snnalexnet":
                        normal_predictions = model.snn(single_image_temp.unsqueeze(1))
                        normal_predictions = decode(normal_predictions)
                time_normal_execution = round(time.time() - time_start, 2)


                #encrypted execution
                time_start = time.time()
                if model_name == "lenet5" or model_name == "alexnet":
                    encr_predictions = encr_model(encr_image, single_image.unsqueeze(0), debug=True)
                elif model_name == "snnlenet5" or model_name == "snnalexnet":
                    encr_predictions = encr_model(encr_image, single_image_temp.unsqueeze(1), debug=True)
                    encr_predictions = decode(encr_predictions)
                time_encr_execution = round(time.time() - time_start, 2)


                #plaintext and encrypted comparison
                difference = normal_predictions - encr_predictions
                normal_predict_label = torch.argmax(normal_predictions).item()
                teleprint("Label predicted for normal execution: %s" %normal_predict_label)
                encr_predict_label = torch.argmax(encr_predictions).item()
                teleprint("Label predicted for encrypted execution: %s" %encr_predict_label)
                teleprint("Time for image encryption: %ss" %time_encryption)
                teleprint("Time for normal execution: %ss" %time_normal_execution)
                teleprint("Time for encrypted execution: %ss" %time_encr_execution)

                #errors
                panda_image = pd.DataFrame(columns=['model', 'dataset', 'image', 'number', 'label', 'm', 'p',
                    'time_encryption', 'time_normal_execution', 'time_encr_execution', 'normal_predictions',
                    'encr_predictions', 'normal_predict_label', 'encr_predict_label', 'conv1_err', 'act1_err',
                    'pool1_err', 'conv2_err', 'act2_err', 'pool2_err', 'flat_err', 'lin1_err', 'act3_err', 'lin2_err',
                    'act4_err', 'lin3_err', 'noise_budget'])

                #saving all data in log
                panda_image.loc[0, 'model'] = model_name
                panda_image.loc[0, 'dataset'] = dataset_name
                panda_image.loc[0, 'image'] = single_image.numpy().tolist()
                panda_image.loc[0, 'number'] = number
                panda_image.loc[0, 'label'] = single_label
                panda_image.loc[0, 'm'] = m
                panda_image.loc[0, 'p'] = p
                panda_image.loc[0, 'time_encryption'] = time_encryption
                panda_image.loc[0, 'time_normal_execution'] = time_normal_execution
                panda_image.loc[0, 'time_encr_execution'] = time_encr_execution
                panda_image.loc[0, 'normal_predictions'] = normal_predictions.numpy().tolist()
                panda_image.loc[0, 'encr_predictions'] = encr_predictions.detach().numpy().tolist()
                panda_image.loc[0, 'normal_predict_label'] = normal_predict_label
                panda_image.loc[0, 'encr_predict_label'] = encr_predict_label
                panda_image.loc[0, 'conv1_err'] = function_global.conv1_err
                panda_image.loc[0, 'act1_err'] = function_global.act1_err
                panda_image.loc[0, 'pool1_err'] = function_global.pool1_err
                panda_image.loc[0, 'conv2_err'] = function_global.conv2_err
                panda_image.loc[0, 'act2_err'] = function_global.act2_err
                panda_image.loc[0, 'pool2_err'] = function_global.pool2_err
                panda_image.loc[0, 'flat_err'] = function_global.flat_err
                panda_image.loc[0, 'lin1_err'] = function_global.lin1_err
                panda_image.loc[0, 'act3_err'] = function_global.act3_err
                panda_image.loc[0, 'lin2_err'] = function_global.lin2_err
                panda_image.loc[0, 'act4_err'] = function_global.act4_err
                panda_image.loc[0, 'lin3_err'] = function_global.lin3_err
                panda_image.loc[0, 'noise_budget'] = function_global.noise_budget
                if image_csv_flag == 1:
                    panda_image.to_csv(logpath + "/%s__" %model_name + "%s__" %dataset_name + "image_%05d__" %number + "label_%s__" %single_label + "m_%05d__" %m + "p_%05d" %p + ".csv", index=None)

else:
    teleprint("The encrypted evaluation will NOT be done")


teleprint("-------------------------")

#saving the panda dataframe
if training_flag == 1:
    panda_csv(panda_log, logpath, file_name, send_dataframe=False, validation_flag=validation_flag)
    teleprint("Dataframe saved in: %s" % logpath)
else:
    teleprint("Without training there is not a dataframe to be saved")

logwriter.close()

#saving the arguments in a file
if len(sys.argv) > 0:
    file_log = open(logpath + "/%s" % (file_name) + ".txt", "a")
    teleprint("Arguments list in the command line:")
    file_log.write("Arguments list in the command line:\n\n")
    for i in range(len(sys.argv)):
        teleprint("==> %s" %sys.argv[i])
        file_log.write("%s\n" %sys.argv[i])
    file_log.close()

#saving and sending last files
log_zip(logpath,file_name)
telegram_bot_file(logpath + ".zip")
try: os.remove(logpath + ".zip") #the file is removed only if exists
except: pass
try: telegram_bot_file(model_path_ext) #the file is sent only if exists
except: pass
telegram_bot_file(os.path.basename(__file__))


#printing the end of the script
teleprint("%s" %(os.path.basename(__file__)), bold = True)

telegram_bot_text("\n========END=========")

### END