'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar to mimic xlua.progress.
'''
import matplotlib
matplotlib.use('Agg')

import os
import sys
import time
import math
import argparse

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

# def hyperparams_override(hyperparams, args):
#     for key in hyperparams.keys():
#         arg_str = 'args.' + 

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 45. # changed from 65 to give more space
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

    return L

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def print_model_params(model,verbose=None):
    if verbose == 'Y' or verbose == 'y':
        # Print the model architecture and parameters
        print('Model architectures:\n{}\n'.format(model))

        print('Parameters and size:')
        for name, param in model.named_parameters():
            print('{}: {}'.format(name, list(param.size())))

    num_params = sum([param.nelement() for param in model.parameters()])

    print('\nTotal number of parameters: {:,}\n'.format(num_params))

    return num_params


def accuracy(output, target, cuda_enabled=True):
    """
    Compute accuracy.

    Args:
        output: [batch_size, 10, 16, 1] The output from DigitCaps layer.
        target: [batch_size] Labels for dataset.

    Returns:
        accuracy (float): The accuracy for a batch.
    """
    batch_size = target.size(0)

    #v_length = torch.sqrt((output**2).sum(dim=2, keepdim=True))
    v_length = torch.sqrt((output.pow(2)).sum(dim=2, keepdim=True))
    softmax_v = softmax(v_length, dim=1)
    assert softmax_v.size() == torch.Size([batch_size, 10, 1, 1])

    _, max_index = softmax_v.max(dim=1)
    assert max_index.size() == torch.Size([batch_size, 1, 1])

    pred = max_index.squeeze() #max_index.view(batch_size)
    assert pred.size() == torch.Size([batch_size])

    if cuda_enabled:
        target = target.cuda()
        pred = pred.cuda()

    correct_pred = torch.eq(target, pred.data) # tensor
    # correct_pred_sum = correct_pred.sum() # scalar. e.g: 6 correct out of 128 images.
    acc = correct_pred.float().mean() # e.g: 6 / 128 = 0.046875

    return acc

def make_loss_and_accuracy_plots(args):

    """
    header_row = 'trainortest,model,activation,optimizer,epoch,batch_idx,num_batches,loss,accuracy,correct,total,batch_loss,lr,batch_accuracy,max_batch_acc,min_batch_acc,total_model_params,epoch_standard_deviation,max_accuracy,random_seed,learning_schedule'
    """

    # make sure to update this if making any changes to columns stored in training/testing info files
    dtypes = {    
    'trainortest': 'object',
    'model': 'object',
    'activation': 'object',
    'optimizer': 'object',
    'epoch': 'int',
    'batch_idx': 'int',
    'num_batches': 'int',
    'loss': 'float32',
    'accuracy': 'float32',
    'correct': 'int',
    'total': 'int',
    'batch_loss': 'float32',
    'lr': 'float32',
    'batch_accuracy': 'float32',
    'max_batch_acc': 'float32',
    'min_batch_acc': 'float32',
    'total_model_params': 'int',
    'epoch_standard_devation': 'float32',
    'max_accuracy': 'float32',
    'random_seed': 'int',
    'learning_schedule': 'object'
    }

    accplot_file_name = os.path.join(args.plot_folder,'accplot_'+str(args.current_model)+args.timestamp)
    lossplot_file_name = os.path.join(args.plot_folder,'lossplot_'+str(args.current_model)+args.timestamp)
    trainfile = pd.read_csv(args.training_info_filename,skiprows=1,dtype=dtypes)
    testfile = pd.read_csv(args.testing_info_filename,skiprows=2,dtype=dtypes)
    trainfile = trainfile.groupby('epoch').mean()
    testfile = testfile.groupby('epoch').mean()

    # save a plot of train and test accuracy
    fig, ax1 = plt.subplots(figsize=(16,8))
    plt.title(str(args.current_model))
    ax1.plot(trainfile.index,trainfile.accuracy,c='red',lw=2,label='train accuracy')
    ax1.plot(testfile.index,testfile.accuracy,c='blue',lw=2,label='test accuracy')
    plt.legend(loc='lower right')

    ax2 = ax1.twinx()
    ax2.plot(testfile.index,testfile.lr,c='green',lw=3,label='learning rate')
    plt.legend(loc='lower center')

    plt.savefig(accplot_file_name + '.png')

    # save a plot of train and test loss
    fig, ax1 = plt.subplots(figsize=(16,8))
    plt.title(str(args.current_model))
    ax1.plot(trainfile.index,trainfile.loss,c='red',lw=2,label='train loss')
    ax1.plot(testfile.index,testfile.loss,c='blue',lw=2,label='test loss')
    plt.legend(loc='upper right')

    ax2 = ax1.twinx()
    ax2.plot(testfile.index,testfile.lr,c='green',lw=3,label='learning rate')
    plt.legend(loc='upper center')

    plt.savefig(lossplot_file_name + '.png')
    plt.close('all')

def setup_combo_results(args):

    combo_header_row = 'model,activation,optimizer,trainortest,epoch,batch_idx,num_batches,loss,accuracy,correct,total,batch_loss,lr,batch_accuracy,max_batch_acc,min_batch_acc,total_model_params,epoch_standard_deviation,max_accuracy,random_seed,learning_schedule'

    combo_train_folder = os.path.join(args.training_info_dir, "combo_train")
    combo_test_folder = os.path.join(args.training_info_dir, "combo_test")
    train_summary_folder = os.path.join(args.training_info_dir, "train_summary")
    test_summary_folder = os.path.join(args.training_info_dir, "test_summary")

    hyperparams_folder = os.path.join(args.training_info_dir, "hyperparams")

    if not os.path.isdir(combo_train_folder):
        os.makedirs(combo_train_folder)
    if not os.path.isdir(combo_test_folder):
        os.makedirs(combo_test_folder)

    if not os.path.isdir(train_summary_folder):
        os.makedirs(train_summary_folder)
    if not os.path.isdir(test_summary_folder):
        os.makedirs(test_summary_folder)

    if not os.path.isdir(hyperparams_folder):
        os.makedirs(hyperparams_folder)        

    combo_train_filename = "combo_train_" + str(args.timestamp) + '.csv'
    combo_train_filename = os.path.join(combo_train_folder,combo_train_filename)

    hyperparams_filename = "hyperparams" + '_' + str(args.timestamp) + '.py'
    hyperparams_filename = os.path.join(hyperparams_folder,hyperparams_filename)    

    args.combotrainfile = open(combo_train_filename,"w",encoding="utf-8")
    args.combotrainfile.write(str(args.model_list)+'\n\n')
    args.combotrainfile.write(str(args)+'\n\n')
    args.combotrainfile.write(combo_header_row + '\n')

    combo_test_filename = "combo_test_" + "_".join([str(x) for x in time.localtime()[0:5]]) + '.csv'
    combo_test_filename = os.path.join(combo_test_folder,combo_test_filename)

    args.combotestfile = open(combo_test_filename,"w",encoding="utf-8")
    args.combotestfile.write('\n\n'+str(args.model_list)+'\n\n')
    args.combotestfile.write(str(args)+'\n\n')
    args.combotestfile.write(combo_header_row + '\n')

    train_summary_filename = "train_summary_" + "_".join([str(x) for x in time.localtime()[0:5]]) + '.csv'
    train_summary_filename = os.path.join(train_summary_folder,train_summary_filename)

    args.trainsummaryfile = open(train_summary_filename,"w",encoding="utf-8")
    args.trainsummaryfile.write(str(args.model_list)+'\n\n')
    args.trainsummaryfile.write(str(args)+'\n\n')

    test_summary_filename = "test_summary_" + "_".join([str(x) for x in time.localtime()[0:5]]) + '.csv'
    test_summary_filename = os.path.join(test_summary_folder,test_summary_filename)

    args.testsummaryfile = open(test_summary_filename,"w",encoding="utf-8")
    args.testsummaryfile.write('\n\n'+str(args.model_list)+'\n\n')
    args.testsummaryfile.write(str(args)+'\n\n')

    with open(hyperparams_filename,"w",encoding="utf-8") as f:
        f.write(str(hyperparams))
        f.write('\n\n')
        f.write(str(args))

    return args

def save_training_info(args):

    if not os.path.isdir(args.train_folder):
        os.makedirs(args.train_folder)

    if not os.path.isdir(args.test_folder):
        os.makedirs(args.test_folder)

    if not os.path.isdir(args.plot_folder):
        os.makedirs(args.plot_folder)

    args.training_info_filename = "train" + str(args.timestamp) + "_" + str(args.current_model) + '_' + str(args.current_activation) + "_" + str(args.optimizer) + '.csv'
    args.training_info_filename = os.path.join(args.train_folder,args.training_info_filename)

    args.testing_info_filename = "test" + str(args.timestamp) + "_" + str(args.current_model) + "_" + str(args.current_activation) + "_" + str(args.optimizer) + '.csv'
    args.testing_info_filename = os.path.join(args.test_folder,args.testing_info_filename)        

    args.trainfile = open(args.training_info_filename,"w",encoding="utf-8")
    args.testfile = open(args.testing_info_filename,"w",encoding="utf-8")

    header_row = 'trainortest,model,activation,optimizer,epoch,batch_idx,num_batches,loss,accuracy,correct,total,batch_loss,lr,batch_accuracy,max_batch_acc,min_batch_acc,total_model_params,epoch_standard_deviation,max_accuracy,random_seed,learning_schedule'

    args.trainfile.write(str(args.current_model) + ',' + str(args.current_activation) + ',' + str(args.optimizer) + '\n\n')
    args.trainfile.write(header_row + '\n')

    args.testfile.write('\n' + str(args.current_model) + ',' + str(args.current_activation) + ',' + str(args.optimizer) + '\n')
    args.testfile.write(header_row + '\n')

    args.trainsummaryfile.write('\n'+ str(args.current_model) + ',' + str(args.current_activation) + ',' + str(args.optimizer) + '\n')
    args.trainsummaryfile.write(header_row + '\n')

    args.testsummaryfile.write('\n'+ str(args.current_model) + ',' + str(args.current_activation) + ',' + str(args.optimizer) + '\n')
    args.testsummaryfile.write(header_row + '\n')

    return args

