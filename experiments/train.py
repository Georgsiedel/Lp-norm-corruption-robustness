from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import importlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
#os.chdir('C:\\Users\\Admin\\Desktop\\Python\\corruption-testing')
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(0)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = False

from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from experiments.network import WideResNet
import torchvision.models as models
from experiments.sample_corrupted_img import sample_lp_corr
from torchvision.transforms.autoaugment import AugMix
from torchvision.models import VisionTransformer

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with perturbations')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noise', default='gaussian', type=str, help='type of noise')
parser.add_argument('--epsilon', default=0.1, type=float, help='perturbation radius')
parser.add_argument('--epochs', default=30, type=int, help="number of epochs")
parser.add_argument('--run', default=0, type=int, help='run number')
parser.add_argument('--max', type=str2bool, nargs='?', const=True, default=False, help='sample max epsilon values only (True) or random values up to max epsilon (False)')
parser.add_argument('--experiment', default=0, type=int, help='experiment number - each experiment is defined in module config{experiment}')
args = parser.parse_args()

configname = (f'experiments.configs.config{args.experiment}')
config = importlib.import_module(configname)

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
criterion = nn.CrossEntropyLoss()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Bounds without normalization of inputs
x_min = torch.tensor([0, 0, 0]).to(device).view([1, -1, 1, 1])
x_max = torch.tensor([1, 1, 1]).to(device).view([1, -1, 1, 1])
def train(pbar):
    """ Perform epoch of training"""
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        inputs_pert = inputs

        if config.combine_train_corruptions == True:
            for id, img in enumerate(inputs):
                corruptions_list = random.sample(list(config.train_corruptions), k=config.concurrent_combinations)
                for x, (noise_type, train_epsilon, max) in enumerate(corruptions_list):
                    train_epsilon = float(train_epsilon)
                    if max == 'False':
                        img = sample_lp_corr(noise_type, train_epsilon, img, 'other')
                    elif max == 'True':
                        img = sample_lp_corr(noise_type, train_epsilon, img, 'max')
                inputs_pert[id] = img
        else:
            for id, img in enumerate(inputs):
                if args.max == False:
                    inputs_pert[id] = sample_lp_corr(args.noise, args.epsilon, img, 'other')
                elif args.max == True:
                    inputs_pert[id] = sample_lp_corr(args.noise, args.epsilon, img, 'max')

        inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
        targets_pert = targets
        targets_pert_pred = net(inputs_pert)

        loss = criterion(targets_pert_pred, targets_pert)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = targets_pert_pred.max(1)
        total += targets_pert.size(0)
        correct += predicted.eq(targets_pert).sum().item()

        pbar.set_description('[Train] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(train_loss / (batch_idx + 1),
                                                                                 100. * correct / total,
                                                                                 correct, total))
        pbar.update(1)
        torch.cuda.empty_cache()

    train_acc = 100.*correct/total
    return train_acc

def valid(pbar):
    """ Test current network on validation set"""
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(validationloader):
        inputs_pert = inputs

        inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
        targets_pert = targets
        targets_pert_pred = net(inputs_pert)

        loss = criterion(targets_pert_pred, targets_pert)
        test_loss += loss.item()
        _, predicted = targets_pert_pred.max(1)
        total += targets_pert.size(0)
        correct += predicted.eq(targets_pert).sum().item()

        pbar.set_description(
            '[Test] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(test_loss / (batch_idx + 1), 100. * correct / total,
                                                               correct, total))
        pbar.update(1)
        torch.cuda.empty_cache()
    acc = 100. * correct / total
    return acc

if __name__ == '__main__':
    # Load and transform data
    print('Preparing data..')
    if config.aug_strat_check == True:
        if config.train_aug_strat == 'AugMix': #this needed manual implementation due to Pytorch problems.
            transform_train = transforms.Compose([
                AugMix(), #Normally, AugMix is provided as a transforms function in torchivion
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                #transforms.Resize(64), #only if using ViT on small datasets
                transforms.RandomHorizontalFlip(),
            ])
        else:
            transform_strategy = getattr(transforms, config.train_aug_strat)
            transform_train = transforms.Compose([
                transform_strategy(),
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                #transforms.Resize(64), #only if using ViT on small datasets
                transforms.RandomHorizontalFlip(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            #transforms.Resize(64), #only if using ViT on small datasets
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    transform_test = transforms.Compose([
        #transforms.Resize(64), #only if using ViT on small datasets
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./experiments/data', train=True,
                                                                  download=True,
                                                                  transform=transform_train)
    if config.validontest == True:
        validset = torchvision.datasets.CIFAR10(root='./experiments/data', train=False, download=True,
                                            transform=transform_test)
    else:
        validset = torchvision.datasets.CIFAR10(root='./experiments/data', train=True, download=True,
                                            transform=transform_test)
        # generate indices: instead of the actual data we pass in integers instead
        train_indices, val_indices, _, _ = train_test_split(
            range(len(trainset)),
            trainset.targets,
            stratify=trainset.targets,
            test_size=0.2,
            random_state=1 #args.run #this is important to have the same validation split when calling train multiple times
        )
        # generate subset based on indices

        trainset = Subset(trainset, train_indices)
        validset = Subset(validset, val_indices)

    if config.train_aug_strat == True:
        print(config.train_aug_strat, 'is used')

    # create batches
    batchsize_train = config.batchsize
    batchsize_valid = config.batchsize
    trainloader = DataLoader(trainset, batch_size=batchsize_train, shuffle=True)    #, num_workers=2)
    validationloader = DataLoader(validset, batch_size=batchsize_valid, shuffle=True) #, num_workers=2)
    # Construct model
    print('\nBuilding', config.modeltype, 'model')
    if config.modeltype == 'wrn28':
        net = WideResNet(28, 10, 0.3, 10)
    else:
        torchmodel = getattr(models, config.modeltype)
        net = torchmodel(**config.modelspecs)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('\nResuming from checkpoint..')
        if not config.combine_train_corruptions:
            checkpoint = torch.load(f'./experiments/models/{config.modeltype}/{args.noise}/{config.modeltype}_epsilon_{args.epsilon}_run_{args.run}.pth')
        else:
            checkpoint = torch.load(f'./experiments/models/{config.modeltype}/{config.modeltype}_config{args.experiment}_concurrent_{config.concurrent_combinations}_run_{args.run}.pth')
            
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1

    # Number of batches
    # NOTE: It's (40,000 + 10,000) / 32
    if config.validontest == True:
        total_steps = (50000/batchsize_train + 10000/batchsize_valid) * args.epochs
    elif config.validontest == False:
        total_steps = (40000/batchsize_train + 10000/batchsize_valid) * args.epochs

    opti = getattr(optim, config.optimizer)
    optimizer = opti(net.parameters(), lr=args.lr, **config.optimizerparams)

    schedule = getattr(optim.lr_scheduler, config.lrschedule)
    scheduler = schedule(optimizer, **config.lrparams)
    train_accs = []
    valid_accs = []
    # Training loop
    print('\nTraining model..')
    with tqdm(total=total_steps) as pbar:
        for epoch in range(start_epoch, start_epoch+args.epochs):
            train_acc = train(pbar)
            valid_acc = valid(pbar)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)

            if config.lrschedule == 'ReduceLROnPlateau':
                scheduler.step(valid_acc)
            else:
                scheduler.step()

        # Save final epoch
        state = {
            'net': net.state_dict(),
            'acc': valid_acc,
            'train_acc': train_acc,
            'epoch': start_epoch+args.epochs-1,
        }
        if config.combine_train_corruptions == True:
            torch.save(state, f'./experiments/models/{config.modeltype}/{config.modeltype}_config{args.experiment}_concurrent_{config.concurrent_combinations}_run_{args.run}.pth')
        else:
            torch.save(state, f'./experiments/models/{config.modeltype}/{args.noise}/{config.modeltype}_epsilon_{args.epsilon}_run_{args.run}.pth')

        print("Maximum test accuracy of", max(valid_accs), "achieved after", np.argmax(valid_accs)+1, "epochs")

        if config.combine_train_corruptions:
            if args.resume:
                old_train_accs = np.loadtxt(f'results/{config.modeltype}/{config.modeltype}_config{args.experiment}_learning_curve_train_concurrent_{config.concurrent_combinations}_run_{args.run}.csv', delimiter=';')
                train_accs = np.append(old_train_accs, train_accs)
                old_valid_accs = np.loadtxt(f'results/{config.modeltype}/{config.modeltype}_config{args.experiment}_learning_curve_valid_concurrent_{config.concurrent_combinations}_run_{args.run}.csv', delimiter=';')
                valid_accs = np.append(old_valid_accs, valid_accs)
            np.savetxt(f'results/{config.modeltype}/{config.modeltype}_config{args.experiment}_learning_curve_train_concurrent_{config.concurrent_combinations}_run_{args.run}.csv', train_accs, delimiter=';')
            np.savetxt(f'results/{config.modeltype}/{config.modeltype}_config{args.experiment}_learning_curve_valid_concurrent_{config.concurrent_combinations}_run_{args.run}.csv', valid_accs, delimiter=';')
        else:
            if args.resume:
                old_train_accs = np.loadtxt(f'results/{config.modeltype}/{config.modeltype}_config{args.experiment}_learning_curve_train_{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.csv', delimiter=';')
                train_accs = np.append(old_train_accs, train_accs)
                old_valid_accs = np.loadtxt(f'results/{config.modeltype}/{config.modeltype}_config{args.experiment}_learning_curve_valid_{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.csv', delimiter=';')
                valid_accs = np.append(old_valid_accs, valid_accs)
            np.savetxt(f'results/{config.modeltype}/{config.modeltype}_config{args.experiment}_learning_curve_train_{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.csv', train_accs, delimiter=';')
            np.savetxt(f'results/{config.modeltype}/{config.modeltype}_config{args.experiment}_learning_curve_valid_{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.csv', valid_accs, delimiter=';')

        x = list(range(1, len(train_accs) + 1))
        plt.plot(x, train_accs, label='Train Accuracy')
        plt.plot(x, valid_accs, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.xticks(np.arange(0, len(train_accs) + 1, (len(train_accs)) / 10))
        plt.legend(loc='best')
        if config.combine_train_corruptions:
            plt.savefig(f'results/{config.modeltype}/{config.modeltype}_config{args.experiment}_learning_curve_concurrent_{config.concurrent_combinations}_run_{args.run}.svg')
        else:
            plt.savefig(f'results/{config.modeltype}/{config.modeltype}_config{args.experiment}_learning_curve_{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.svg')
