################################################################
#################### General imports ###########################
################################################################
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import time
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from functools import partial

################################################################
###################### PyTorch Imports ###########################
################################################################
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

##################################################################
############ Imports that are part of this project ###############
##################################################################

from models import *
from hyperparams import *
from new_optimizers import *
from traintest import train, test
from training_loop_core import train_core

import utils
from utils import progress_bar, print_model_params
from utils import make_loss_and_accuracy_plots, setup_combo_results, save_training_info
from learning_rate_utils import update_lr, decreasing_cycle_lr_update, list_cycle_lr_update, lr_reset

from data_utils import get_cifar10, get_dataset
from imageprocessing import get_transform

###################################################################
################# argparser for training options ##################
###################################################################

# Note that I have overloaded the argparser to hold what are essentially 
# global variables as well as parsing command line arguments.
# Also note that, as a rule, if any of these hyperparameters exist in 
# the `hyperparams.py` file, then those settings will override these
# command-line defaults and inputs.

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--data-path', type=str, default='../data',help="path to store data in")
parser.add_argument('--train-batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=100)
parser.add_argument('--seed', default=999, type=int, help='random seed (default: 999)')
parser.add_argument('--best-test-acc', type=int, default=0) # do not change
parser.add_argument('--start-epoch', type=int, default=0) # do not change - starts from epoch 0 or last checkpoint epoch

######## Hardware-related settings #########
parser.add_argument('--use-cuda', default=torch.cuda.is_available(),
                    help='use cuda or not - replace with False if do not want')
parser.add_argument('--gpus', default=None,
                    help='gpus used for training set to specific numbers to specify- e.g 0,1,3, default is to use all available')
parser.add_argument('--threads', type=int, default=4,
                    help='number of threads for data loader to use. default=4')
parser.add_argument('--optimize-cudnn', action='store_true', default=True, 
                    help='change to false for many different short runs, or runs where input sizes are changing constantly')

######## Optimization and learning rate settings for training #########

# main lr settings and variables
parser.add_argument('--optimizer', type=str, default='SGD',
                    help='Optimizer: choose "Adam", "SGD", "RMSprop" etc')
parser.add_argument('--lr-start', type=float, default=0.01,
                    help='starting learning rate. default=0.01')
parser.add_argument('--lr', type=float, default=None,
                    help='learning rate. set this to a value to override the optimize learning rates in hyperparams file')
parser.add_argument('--lr-floor', type = float, default=0.0001, 
                    help='learning rate floor = minimum learning rate')
parser.add_argument('--lr-change-n-epoch', type=int, default=1,
                    help='adjust learning rate every n epochs. used in most of the learning rate schedules')

# in-epoch lr decay
parser.add_argument('--inepoch-lr-decay', action='store_true', default=False,
                    help='using this decay lr each batch from max to min within each epoch')
parser.add_argument('--inepoch-max', type=float, default=0.32,
                    help='starting point (and max) lr within each epoch')
parser.add_argument('--inepoch-min', type=float, default=0.001,
                    help='ending point (and min) lr within each epoch')
parser.add_argument('--inepoch-max-decay', type=float, default=None,
                    help='optional decay factor for inepoch-max applied after each epoch')
parser.add_argument('--inepoch-min-decay', type=float, default=None,
                    help='optional decay factor for inepoch-max applied after each epoch')
parser.add_argument('--inepoch-batch-step', type=int, default=None, # i.e. 2, or 5 or something
                    help='optional. how many batch iterations to wait before applying lr change')

# decreasing cycle lr
parser.add_argument('--decreasing-cycle-lr', action='store_true', default=False,
                    help='use decreasing cyclic method to update learning rates')
parser.add_argument('--lr-change-increment', type=float, default=0.2,
                    help='used with decreasing_cycle_lr - the percentage increment for decreasing_cycle_lr')
parser.add_argument('--lr-trend-adjustment-factor', type=float, default=.97,
                    help='used with decreasing_cycle_lr - multiplied by lr each epoch')
parser.add_argument('--upcycle', default=False,
                    help='used with decreasing_cycle_lr - starts on False to decrease lr first before increasing')

# cycle lr list
parser.add_argument('--use-cycle-lr-list', action='store_true', default=False,
                    help='using this argument will evaluate the list-string set in the cycle-lr-list argument')
parser.add_argument('--lr-cycle-list-repeat', action='store_true', default=False,
                    help='repeat the list once you get to the end. to use must set cycle-lr-list in hyperparams file')
parser.add_argument('--cycle-lr-list', type=str, default='[3.0,0.3,0.2,0.1]',
                    help='default list of learning rates to step through - can be set in hyperparams file')

# lr fixed stages
parser.add_argument('--lr-fixed-stages', action='store_true', default=False,
                    help='use fixed learning rate steps')
parser.add_argument('--lr-increment', type=float, default=10.,
                    help='used with --lr-fixed-stages. amount to divide learning rate by. default=10')

# misc lr settings and variables
parser.add_argument('--lr-momentum', type=float, default=None,
                    help='learning rate momentum. default=None, but can use 0.9 etc.')
parser.add_argument('--lr-decay', type = float, default=None, 
                    help='learning rate decay. set to below one for decay (i.e. .99, .95 etc)')
parser.add_argument('--lr_schedule', type=str, default="")

########### Logging and checkpoint settings #############

parser.add_argument('--long-progress-bar', action='store_true', default=False,
                    help='longer more informative progress_bar - use this argument or change to True to get more info if your terminal screen is long enough')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint-file', type=str, default=None, help='which checkpoint file to use (filename)')
parser.add_argument('--save-checkpoints', action='store_true', default=False)
parser.add_argument('--checkpoint-dir', type=str, default='results/checkpoints')
parser.add_argument('--checkpoint-accuracy-threshold', type=float, default=50.1,
                    help='accuracy threshold for saving checkpoints. 0-100, default=50.1')
parser.add_argument('--min-checkpoint-improvement', type=float, default=0.1,
                    help='minimum accuracy improvement before saving next checkpoint')
parser.add_argument('--save-training-info', action='store_true', default=True)
parser.add_argument('--make-plots', action='store_true', default=True)
parser.add_argument('--training-info-dir', type=str, default='results/training_info')
parser.add_argument('--summary-increment', type=int, default=10,
                    help='save a summary line every n epochs')
parser.add_argument('--debug-mode', action='store_true', default=False,
                    help='shows print statements to aid debugging. default=false')
parser.add_argument('--total-model-params', type=str, default=None)
parser.add_argument('--current-model', type=str, default=None)
parser.add_argument('--model-list', type=str, default=None)
parser.add_argument('--trainfile', type=str, default=None)
parser.add_argument('--testfile', type=str, default=None)
parser.add_argument('--combotrainfile', type=str, default=None)
parser.add_argument('--combotestfile', type=str, default=None)
parser.add_argument('--trainsummaryfile', type=str, default=None)
parser.add_argument('--testsummaryfile', type=str, default=None)
parser.add_argument('--training-info-filename', type=str, default=None)
parser.add_argument('--testing-info-filename', type=str, default=None)
parser.add_argument('--timestamp', type=str, default=None)

args = parser.parse_args()

args.orig_lr = args.lr
args.orig_upcycle = args.upcycle
##############################################################################
################# CUDA random seed settings (duplicate) ######################
##############################################################################

torch.manual_seed(args.seed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = args.optimize_cudnn
    torch.cuda.manual_seed_all(args.seed)

#############################################################
################## List of Models To Train ##################
#############################################################
"""
To test a different set of models, activation functions,
optimizers, or random seeds, just change these manually 
in the hyperparms.py file.

Refer to TRAINING_INSTRUCTIONS.md for details.
"""
args.model_list = hyperparams['model_list']
args.activation_list = hyperparams['activation_list']
args.optimizer_list = hyperparams['optimizer_list']
args.random_seed_list = hyperparams['random_seed_list']

args.model_permutations = len(args.model_list) * len(args.activation_list) * len(args.optimizer_list) * len(args.random_seed_list) * hyperparams['num_repeats']
total_epochs = args.model_permutations * args.epochs

print()
print("NOTE THAT YOU ARE TESTING A TOTAL OF {} MODEL PERMUTATIONS".format(args.model_permutations))
print()
print("This will take at least as long as one model with {} epochs.".format(total_epochs))
print()

############################################################
################# Set up files for logging #################
############################################################

args.timestamp = "_".join([str(x) for x in time.localtime()[0:5]])

# Set up combo results files outside the loop
if args.save_training_info:
    args.train_folder = os.path.join(args.training_info_dir,"train")
    args.test_folder = os.path.join(args.training_info_dir,"test")
    args.plot_folder = os.path.join(args.training_info_dir,"plots")

    args = setup_combo_results(args)

###################### Load data ###########################
print('==> Preparing data..')
trainset, testset, trainloader, testloader, classes = get_cifar10(args)

######################################################################################
################### Loop through all models and hyperparameters ######################
######################################################################################

args.permutation_number = 0
    
for activation in args.activation_list:
    args.current_activation = activation

    for op, op_lr in args.optimizer_list:
        args.optimizer = op
        if args.lr:
            args.lr_start = args.lr
        else:
            args.lr_start = op_lr

        for seed in args.random_seed_list:
            args.seed = seed
            args.repeat_number = 0

            for args.model_position, this_model in enumerate(args.model_list):
                args.current_model = this_model

                args.repeat_number = 0
                for i in range(hyperparams['num_repeats']):
                    args = lr_reset(args)
                    args.permutation_number += 1
                    args.repeat_number += 1
                    args, optimizer, net = train_core(args, trainloader, testloader)

# final cleanup outside main training loop after all models are trained
if args.save_training_info:
    args.combotrainfile.close()
    args.combotestfile.close()