import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import progress_bar, print_model_params
from new_optimizers import *

#####################################################
############### Define train function ###############
#####################################################

# Training
def train(epoch, net, criterion, optimizer, trainloader, args):

    print('\nEpoch: %d/%d | Model Architecture: %s (%d/%d) | Activation: %s | Optimizer: %s | Seed: %d | Repeat %d | Total Model Permutations: %d/%d |' 
        % (epoch+1, args.epochs, args.current_model, args.model_position + 1, len(args.model_list)+1, args.current_activation, args.optimizer, args.seed, args.repeat_number, args.permutation_number, args.model_permutations))

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    max_batch_acc = 0
    min_batch_acc = 100
    max_accuracy = 0

    # set up list of lrs for inepoch decay (cyclical LR)
    if args.inepoch_lr_decay:
        if args.inepoch_max_decay:
            args.inepoch_max *= args.inepoch_max_decay
        if args.inepoch_min_decay:
            args.inepoch_min *= args.inepoch_min_decay
        if args.inepoch_batch_step:
            num_inepoch_lrs = (len(trainloader)//args.inepoch_batch_step) + 1
        else:
            num_inepoch_lrs = len(trainloader)

        batch_lrs = list(np.linspace(args.inepoch_min, args.inepoch_max, num=num_inepoch_lrs))

    # create array to store batch data
    this_epoch_batch_accuracies = np.array(())        

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        torch.manual_seed(args.seed)

        if args.use_cuda:
            torch.cuda.manual_seed_all(args.seed)
            inputs, targets = inputs.cuda(), targets.cuda()
            criterion.cuda()

        if args.inepoch_lr_decay:

            if args.inepoch_batch_step:
                if batch_idx % args.inepoch_batch_step == 0:
                    args.lr = batch_lrs.pop()
            else:
                args.lr = batch_lrs.pop()
            
            if args.lr_momentum:
               optimizer = eval(args.optimizer + '(net.parameters(), lr=' + str(args.lr) + ', momentum=' + str(args.lr_momentum) +')' )
            else:
               optimizer = eval(args.optimizer + '(net.parameters(), lr=' + str(args.lr) +')' )

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        batch_correct = predicted.eq(targets.data).cpu().sum()
        batch_accuracy = 100.*batch_correct / args.train_batch_size
        correct += batch_correct
        train_accuracy = 100.*correct/total
        this_epoch_batch_accuracies = np.append(this_epoch_batch_accuracies, batch_accuracy)
        acc_std_dev = this_epoch_batch_accuracies.std()            

        if batch_accuracy > max_batch_acc:
            max_batch_acc = batch_accuracy
        if batch_accuracy < min_batch_acc:
            min_batch_acc = batch_accuracy 

        if train_accuracy > max_accuracy:
            max_accuracy = train_accuracy

        """ copied from utils.save_training_info for reference:
        header_row = 'trainortest,activation,optimizer,epoch,batch_idx,num_batches,loss,accuracy,correct,total,batch_loss,lr,batch_accuracy,max_batch_acc,min_batch_acc,total_model_params,epoch_standard_deviation,max_accuracy,random_seed,learning_schedule'
        """

        all_training_info_line = '%d,%d,%d,%.3f,%.3f,%d,%d,%.3f,%.4f,%.3f,%.3f,%.3f,%d,%.2f,%.2f,%d,%s' \
                                % (epoch, 
                                    batch_idx, 
                                    len(trainloader), 
                                    train_loss/(batch_idx+1), 
                                    train_accuracy, 
                                    correct, 
                                    total, 
                                    loss.data[0], 
                                    optimizer.defaults['lr'],
                                    batch_accuracy, 
                                    max_batch_acc, 
                                    min_batch_acc,
                                    args.total_model_params,
                                    acc_std_dev, 
                                    max_accuracy, 
                                    args.seed, 
                                    args.lr_schedule)

        # start the command line progress bar
        if args.long_progress_bar:

            train_progress_bar_output = progress_bar(batch_idx, len(trainloader), 
                'Loss: %.3f | Acc: %.2f%% (%d/%d) | Batch Loss: %.2f | lr: %.5f | Batch Acc: %.2f%% (Max: %.2f%%, Min: %.2f%%) | Std: %.2f | Max: %.2f%% |'
                                    % (train_loss/(batch_idx+1), train_accuracy, correct, total, loss.data[0], optimizer.defaults['lr'],
                                        batch_accuracy, max_batch_acc, min_batch_acc, acc_std_dev, max_accuracy))

        else:
            train_progress_bar_output = progress_bar(batch_idx, len(trainloader), 
                'Loss: %.3f | Acc: %.2f%% | lr: %.4f | Batch Acc: %.2f%% (Max: %.2f%%, Min: %.2f%%)'
                                    % (train_loss/(batch_idx+1), train_accuracy, optimizer.defaults['lr'],
                                        batch_accuracy, max_batch_acc, min_batch_acc))

        # Save Training Info to CSV file
        if args.save_training_info:
            args.trainfile.write('\n' + 'train,' + str(args.current_model) + ',' + str(args.current_activation) + ',' + str(args.optimizer) + ',' + all_training_info_line + '\n')             
            if batch_idx == len(trainloader)-1 or batch_idx == len(trainloader):
                args.combotrainfile.write(str(args.current_model) + ',' + str(args.current_activation) + ',' + str(args.optimizer) + ',train,' + str(args.current_model) + ',' + str(args.current_activation) + ',' + str(args.optimizer) + ',' + all_training_info_line+'\n')
                if epoch % args.summary_increment == 0:
                    args.trainsummaryfile.write('train,' + str(args.current_model) + ',' + str(args.current_activation) + ',' + str(args.optimizer) + ',' + all_training_info_line+'\n')

    return args, optimizer

####################################################
############### Define test function ###############
####################################################

def test(epoch, net, criterion, optimizer, testloader, args):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    max_batch_acc = 0
    min_batch_acc = 100
    max_accuracy = 0

    # create array to store batch data
    this_epoch_batch_accuracies = np.array(())

    for batch_idx, (inputs, targets) in enumerate(testloader):
        torch.manual_seed(args.seed)
        if args.use_cuda:
            torch.cuda.manual_seed_all(args.seed)
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        batch_correct = predicted.eq(targets.data).cpu().sum()
        batch_accuracy = 100.*batch_correct / args.test_batch_size
        correct += batch_correct
        test_accuracy = 100.*correct/total
        this_epoch_batch_accuracies = np.append(this_epoch_batch_accuracies, batch_accuracy)
        acc_std_dev = this_epoch_batch_accuracies.std()

        if batch_accuracy > max_batch_acc:
            max_batch_acc = batch_accuracy
        if batch_accuracy < min_batch_acc:
            min_batch_acc = batch_accuracy

        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy

        """ copied from utils for reference:
        header_row = 'trainortest,epoch,batch_idx,num_batches,loss,accuracy,correct,total,batch_loss,lr,batch_accuracy,max_batch_acc,min_batch_acc,total_model_params,epoch_standard_deviation,max_accuracy,random_seed,learning_schedule'
        """

        all_testing_info_line = '%d,%d,%d,%.3f,%.3f,%d,%d,%.3f,%.4f,%.3f,%.3f,%.3f,%d,%.2f,%.2f,%d,%s' \
                                % (epoch, batch_idx, len(testloader), test_loss/(batch_idx+1), test_accuracy, correct, total, loss.data[0], optimizer.defaults['lr'],batch_accuracy, max_batch_acc, min_batch_acc,args.total_model_params, acc_std_dev, max_accuracy, args.seed,args.lr_schedule)

        # start the command line progress bar
        if args.long_progress_bar:

            test_progress_bar_output = progress_bar(batch_idx, len(testloader), 
                'Loss: %.3f | Acc: %.2f%% (%d/%d) | Batch Loss: %.2f | lr: %s | Batch Acc: %.2f%% (Max: %.2f%%, Min: %.2f%%) | Std: %.2f | Max: %.2f%% |'
                                    % (test_loss/(batch_idx+1), test_accuracy, correct, total, loss.data[0], args.lr_schedule, batch_accuracy, max_batch_acc, min_batch_acc, acc_std_dev, max_accuracy))

        else:
            test_progress_bar_output = progress_bar(batch_idx, len(testloader), 
                'Loss: %.3f | Acc: %.2f%% | lr: %s | Batch Acc: %.2f%% (Max: %.2f%%, Min: %.2f%%)'
                                    % (test_loss/(batch_idx+1), test_accuracy, args.lr_schedule,
                                        batch_accuracy, max_batch_acc, min_batch_acc))

        # Save Test Info to to the training info CSV file
        if args.save_training_info:
            args.testfile.write('\n' + 'test,' + str(args.current_model) + ',' + str(args.current_activation) + ',' + str(args.optimizer) + ',' + all_testing_info_line+'\n')
            if batch_idx == len(testloader)-1 or batch_idx == len(testloader):
                args.combotestfile.write('test,' + str(args.current_model) + ',' + str(args.current_activation) + ',' + str(args.optimizer) + ',' + all_testing_info_line+'\n')
                if epoch % args.summary_increment == 0:
                    args.testsummaryfile.write('test,' + str(args.current_model) + ',' + str(args.current_activation) + ',' + str(args.optimizer) + ',' + all_testing_info_line+'\n')                    

        # save checkpoints for swa average of params
        if args.swa:
            save_swa_checkpoint(net, args)

        # Save checkpoint.
        if args.save_checkpoints:
            if test_accuracy > args.checkpoint_accuracy_threshold \
                and batch_idx >= args.test_batch_size//1.2 \
                and test_accuracy > args.best_test_acc + args.min_checkpoint_improvement:

                if not os.path.isdir(args.checkpoint_dir):
                    os.makedirs(args.checkpoint_dir)

                checkpoint_filename = str(args.current_model) + '_' + str(args.optimizer) + "_" + str(args.current_activation) + "_" + "acc{:.0f}".format(test_accuracy) + '_ckpt.t' + str(epoch)
                checkpoint_filename = os.path.join(args.checkpoint_dir, checkpoint_filename)
                print('Saving checkpoint {}'.format(checkpoint_filename))

                state = {
                    'net': net.module if use_cuda else net,
                    'acc': test_accuracy,
                    'epoch': epoch,
                }

                torch.save(state, checkpoint_filename)

            if test_accuracy > args.best_test_acc:
                args.best_test_acc = test_accuracy