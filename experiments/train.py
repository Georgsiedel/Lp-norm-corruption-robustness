from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
#os.chdir('C:\\Users\\Admin\\Desktop\\Python\\corruption-testing')
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from experiments.network import WideResNet
from experiments.sample_corrupted_img import sample_lp_corr
from experiments.config import train_corruptions
from experiments.config import combine_train_corruptions
from experiments.config import concurrent_combinations
from experiments.config import train_aug_strat
from experiments.config import aug_strat_check
from torchvision.transforms.autoaugment import AugMix

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with perturbations')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noise', default='gaussian', type=str, help='type of noise')
parser.add_argument('--epsilon', default=0.1, type=float, help='perturbation radius')
parser.add_argument('--epochs', default=30, type=int, help="number of epochs")
parser.add_argument('--run', default=0, type=int, help='run number')
parser.add_argument('--max', default=False, type=bool, help='sample max epsilon values only (True) or random values up to max epsilon (False)')

args = parser.parse_args()

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

        if combine_train_corruptions == True:
            for id, img in enumerate(inputs):
                corruptions_list = random.sample(list(train_corruptions), k=concurrent_combinations)
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

    acc = 100. * correct / total
    return acc

if __name__ == '__main__':
    # Load and transform data
    print('Preparing data..')
    if aug_strat_check == True:
        if train_aug_strat == 'AugMix': #this needed manual implementation due to Pytorch problems.
            transform_augmentation_strategy = transforms.Compose([
                AugMix(), #Normally, AugMix is provided as a transforms function in torchivion
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            transform_strategy = getattr(transforms, train_aug_strat)
            transform_augmentation_strategy = transforms.Compose([
                transform_strategy(),
                transforms.ToTensor(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        trainset_augmentation_strategy = torchvision.datasets.CIFAR10(root='./experiments/data', train=True,
                                                                      download=True,
                                                                      transform=transform_augmentation_strategy)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset_transformed = torchvision.datasets.CIFAR10(root='./experiments/data', train=True, download=True,
                                                        transform=transform_train)
    trainset = torchvision.datasets.CIFAR10(root='./experiments/data', train=True, download=True,
                                            transform=transform_test)

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, val_indices, _, _ = train_test_split(
        range(len(trainset)),
        trainset.targets,
        stratify=trainset.targets,
        test_size=0.2,
        random_state=1 #this is important to have the same validation split when calling train multiple times
    )
    # generate subset based on indices
    if aug_strat_check == True:
        train_split = Subset(trainset_augmentation_strategy, train_indices)
        val_split = Subset(trainset, val_indices)
        print(train_aug_strat, 'is used')
    else:
        train_split = Subset(trainset_transformed, train_indices)
        val_split = Subset(trainset, val_indices)
    # create batches
    trainloader = DataLoader(train_split, batch_size=32, shuffle=True)    #, num_workers=2)
    validationloader = DataLoader(val_split, batch_size=32, shuffle=True) #, num_workers=2)
    # Construct model
    print('\nBuilding model..')
    net = WideResNet(28, 10, 0.3, 10)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('\nResuming from checkpoint..')
        if not combine_train_corruptions:
            checkpoint = torch.load(f'./experiments/models/{args.noise}/cifar_epsilon_{args.epsilon}_run_{args.run}.pth')
        else:
            checkpoint = torch.load(f'./experiments/models/cifar_combined_0_concurrent_{concurrent_combinations}_run_{args.run}.pth')
            
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1

    # Number of batches
    # NOTE: It's (40,000 + 10,000) / 32
    total_steps = 1563 * args.epochs
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
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

        # Save final epoch
        state = {
            'net': net.state_dict(),
            'acc': valid_acc,
            'train_acc': train_acc,
            'epoch': start_epoch+args.epochs-1,
        }
        if combine_train_corruptions == True:
            torch.save(state, f'./experiments/models/cifar_combined_0_concurrent_{concurrent_combinations}_run_{args.run}.pth')
        else:
            torch.save(state, f'./experiments/models/{args.noise}/cifar_epsilon_{args.epsilon}_run_{args.run}.pth')

        print("Maximum test accuracy of", max(valid_accs), "achieved after", np.argmax(valid_accs)+1, "epochs")

        if combine_train_corruptions:
            if args.resume:
                old_train_accs = np.loadtxt(f'results/learning_curve_train_combined_0_concurrent_{concurrent_combinations}_run_{args.run}.csv', delimiter=';')
                train_accs = np.append(old_train_accs, train_accs)
                old_valid_accs = np.loadtxt(f'results/learning_curve_valid_combined_0_concurrent_{concurrent_combinations}_run_{args.run}.csv', delimiter=';')
                valid_accs = np.append(old_valid_accs, valid_accs)
            np.savetxt(f'results/learning_curve_train_combined_0_concurrent_{concurrent_combinations}_run_{args.run}.csv', train_accs, delimiter=';')
            np.savetxt(f'results/learning_curve_valid_combined_0_concurrent_{concurrent_combinations}_run_{args.run}.csv', valid_accs, delimiter=';')
        else:
            if args.resume:
                old_train_accs = np.loadtxt(f'results/learning_curve_train_{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.csv', delimiter=';')
                train_accs = np.append(old_train_accs, train_accs)
                old_valid_accs = np.loadtxt(f'results/learning_curve_valid_{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.csv', delimiter=';')
                valid_accs = np.append(old_valid_accs, valid_accs)
            np.savetxt(f'results/learning_curve_train_{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.csv', train_accs, delimiter=';')
            np.savetxt(f'results/learning_curve_valid_{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.csv', valid_accs, delimiter=';')

        x = list(range(1, len(train_accs) + 1))
        plt.plot(x, train_accs, label='Train Accuracy')
        plt.plot(x, valid_accs, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.xticks(np.arange(0, len(train_accs) + 1, (len(train_accs)) / 10))
        plt.legend(loc='best')
        if combine_train_corruptions:
            plt.savefig(f'results/learning_curve_combined_0_concurrent_{concurrent_combinations}_run_{args.run}.svg')
        else:
            plt.savefig(f'results/learning_curve_{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.svg')