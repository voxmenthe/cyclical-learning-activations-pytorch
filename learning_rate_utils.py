import os
import sys
import time
import math
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets
import torchvision.utils as vutils

from new_optimizers import *
from hyperparams import *

def update_lr(epoch, args, net, optimizer):
    if args.lr_decay and args.lr_fixed_stages:
        print('\n\n')
        print("Learning decay and learning stages cannot be set simultaneously.")
        print("Learning rate will not be updated.")
        print('\n\n')

    if args.lr_decay and not args.lr_fixed_stages:
        if args.lr > args.lr_floor:
            args.lr *= args.lr_decay
            if args.lr_momentum:
                optimizer = eval(str(args.optimizer) + '(net.parameters(), lr=' + str(args.lr) + \
                    ', momentum=' + str(args.lr_momentum) +')' )
            else:
                optimizer = eval(str(args.optimizer) + '(net.parameters(), lr=' + str(args.lr) +')' )

    if args.lr_fixed_stages:
        if args.lr > args.lr_floor:
            if epoch % args.lr_change_n_epoch==0:
                args.lr /= args.lr_increment
                
                if args.lr_momentum:
                    optimizer = eval(str(args.optimizer) + '(net.parameters(), lr=' + str(args.lr) + \
                        ', momentum=' + str(args.lr_momentum) +')')
                else:
                    optimizer = eval(str(args.optimizer) + '(net.parameters(), lr=' + str(args.lr) +')' )

    return args, net, optimizer


def decreasing_cycle_lr_update(epoch, args, net, optimizer):

    # at a later stage - add in noise to the lr once you've got it working - make optional
    # I think it's fine for trend multiplication to come after change increment, but can think about further later

    if args.lr > args.lr_floor and epoch % args.lr_change_n_epoch==0:
        if args.upcycle:
            args.lr *= (1 + args.lr_change_increment)
            args.lr *= args.lr_trend_adjustment_factor
            args.upcycle = False
        else:
            args.lr *= (1-args.lr_change_increment)
            args.lr *= args.lr_trend_adjustment_factor 
            if args.lr < args.lr_floor:
                args.lr = args.lr_floor
            args.upcycle = True

        if args.lr_momentum:
            optimizer = eval(str(args.optimizer) + '(net.parameters(), lr=' + str(args.lr) + \
                ', momentum=' + str(args.lr_momentum) +')')
        else:
            optimizer = eval(str(args.optimizer) + '(net.parameters(), lr=' + str(args.lr) +')' )

    return args, net, optimizer

def list_cycle_lr_update(epoch, args, net, optimizer):
    # maybe do another version w/ change factors

    if args.lr_cycle_list_repeat and not args.cycle_lr_list:
        args.cycle_lr_list = eval(hyperparams['cycle_lr_list'])

    # commented out since using lr_change_n_epoch instead
    #if args.cycle_lr_list and args.lr > args.lr_floor and epoch % args.epoch_increment==0:
    if args.cycle_lr_list and args.lr > args.lr_floor and epoch % args.lr_change_n_epoch==0:
        args.lr = args.cycle_lr_list.pop(0)
        #args.epoch_threshold += args.epoch_increment

    if args.lr_momentum:
        optimizer = eval(str(args.optimizer) + '(net.parameters(), lr=' + str(args.lr) + \
            ', momentum=' + str(args.lr_momentum) +')')
    else:
        optimizer = eval(str(args.optimizer) + '(net.parameters(), lr=' + str(args.lr) +')' )

    return args, net, optimizer

def lr_reset(args):
    args.lr = args.orig_lr
    args.upcycle = args.orig_upcycle
    return args
