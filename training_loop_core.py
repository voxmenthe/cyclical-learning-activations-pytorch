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
from new_optimizers import *
from hyperparams import *
from traintest import train, test

import utils
from utils import progress_bar, print_model_params
from utils import make_loss_and_accuracy_plots, setup_combo_results, save_training_info
from learning_rate_utils import update_lr, decreasing_cycle_lr_update, list_cycle_lr_update

from data_utils import get_cifar10, get_dataset
from imageprocessing import get_transform

def train_core(args, trainloader, testloader):
	args.best_test_acc = 0  # best test accuracy
	args.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
	args.lr = args.lr_start

	# Model
	print('\n==> Building model: {} with activation {} and optimizer {} and random seed {}.'.format(args.current_model, args.current_activation, args.optimizer, args.seed))
	print('>>This is model architecture #{} of {} and model permutation #{} out of a total of {} permutations.'.format(args.model_position+1, len(args.model_list)+1, args.permutation_number, args.model_permutations))

	try:
		model_string = str(args.current_model) + '(activation=' + str(args.current_activation) + ')'
		net = eval(model_string)
	except:
		print("This model using default activation only")
		self.activation='Disabled'
		model_string = str(args.current_model) + '()'
		net = eval(model_string)

	# print model parameters using function from utils
	args.total_model_params = print_model_params(net)

	################################################################
	###################### CUDA settings ###########################
	################################################################

	torch.manual_seed(args.seed)

	if args.use_cuda:
	   torch.cuda.manual_seed_all(args.seed)
	   net.cuda()

	   if args.gpus:
	       args.gpus = [int(i) for i in args.gpus.split(',')]
	       torch.cuda.set_device(args.gpus[0])
	   else:
	       net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

	   cudnn.deterministic = True
	   cudnn.benchmark = args.optimize_cudnn
	   torch.cuda.manual_seed_all(args.seed)

	#######################################################
	############# Loss and Optimizer Settings #############
	#######################################################

	criterion = nn.CrossEntropyLoss()

	# Initialize the learning rate settings

	if args.lr_fixed_stages:
		print("Using lr_fixed_stages method to update learning rates")
		args.epoch_increment = args.epochs // args.lr_stages
		args.epoch_threshold = args.epoch_increment
		args.lr_schedule = "Stages" + "_" + str(args.lr_stages)

	elif args.use_cycle_lr_list:
		print("Using cycle_lr_list method to update learning rates")

		# over-write defaults with settings from hyperparams if available
		if hyperparams['cycle_lr_list']:
			args.cycle_lr_list = hyperparams['cycle_lr_list']
		args.cycle_lr_list = eval(args.cycle_lr_list)
		# commented out since should use lr_change_n_epoch instead
		# args.epoch_increment = args.epochs // len(args.cycle_lr_list)
		# args.epoch_threshold = args.epoch_increment
		args.lr_schedule = "CycleList" #+ '_' + str(args.cycle_lr_list)
		print('Using lr cycle list: ', args.cycle_lr_list)
		args.lr = args.cycle_lr_list.pop(0)

	elif args.decreasing_cycle_lr:
		print("Using decreasing cycle method to update learning rates")
		args.lr_schedule = "DecreaseCycle"

	elif args.lr_decay:
		print("Using learing rate decay to update learning rates")
		args.lr_schedule = "LRDecay"

	elif args.inepoch_lr_decay:

		# over-write defaults with settings from hyperparams if available
		if hyperparams['inepoch_max']:
			args.inepoch_max = hyperparams['inepoch_max']
		if hyperparams['inepoch_min']:
			args.inepoch_min = hyperparams['inepoch_min']
		if hyperparams['inepoch_max_decay']:
			args.inepoch_max_decay = hyperparams['inepoch_max_decay']
		if hyperparams['inepoch_min_decay']:
			args.inepoch_min_decay = hyperparams['inepoch_min_decay']
		if hyperparams['inepoch_batch_step']:
			args.inepoch_batch_step = hyperparams['inepoch_batch_step']

		print("Using in-batch learning rate decay to train with max {} and min {}".format(args.inepoch_max,args.inepoch_min))
		print("Max decay rate: {}  Min decay rate {}".format(args.inepoch_max_decay, args.inepoch_min_decay))
		args.lr_schedule = "InEpochDecay"

	if args.lr_momentum:
	   optimizer = eval(args.optimizer + '(net.parameters(), lr=' + str(args.lr) + ', momentum=' + str(args.lr_momentum) +')' )
	else:
	   optimizer = eval(args.optimizer + '(net.parameters(), lr=' + str(args.lr) +')' )

	#######################################################################
	############### Set up files to save training info into ###############
	#######################################################################

	# Prepare to save training info to csv file
	if args.save_training_info:
	   args = save_training_info(args)

	##################################################
	############### Main training loop ###############
	##################################################

	for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
	   args, optimizer = train(epoch, net, criterion, optimizer, trainloader, args)
	   test(epoch, net, criterion, optimizer, testloader, args)

	   ################################################################
	   ################## Updating the learning rate ##################
	   ################################################################

	   if args.decreasing_cycle_lr:
	   	args, net, optimizer = decreasing_cycle_lr_update(epoch, args, net, optimizer)
	   elif args.use_cycle_lr_list:
	   	args, net, optimizer = list_cycle_lr_update(epoch, args, net, optimizer)
	   else:
	   	args, net, optimizer = update_lr(epoch, args, net, optimizer)

	if args.save_training_info:
	   args.trainfile.close()
	   args.testfile.close()

	if args.make_plots:
	   make_loss_and_accuracy_plots(args)

	return args, optimizer, net
