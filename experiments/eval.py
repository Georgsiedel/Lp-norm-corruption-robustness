from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import os
#os.chdir('C:\\Users\\Admin\\Desktop\\Python\\corruption-testing')

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
from skimage.util import random_noise
import torch.distributions as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from experiments.network import WideResNet
from experiments.sample_corrupted_img import sample_lp_corr

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch Evaluation with perturbations')
parser.add_argument('--test_on_c', type=str2bool, nargs='?', const=True, default=True, help='Whether to test on the C-benchmark by Hendrycks2019')
parser.add_argument('--combine_test_corruptions', type=str2bool, nargs='?', const=True, default=False, help='Whether to combine all test noise values by drawing from the randomly')
parser.add_argument('--batchsize', default=128, type=int, help='Images per batch - more means quicker training, but higher memory demand')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noise', default='gaussian', type=str, help='type of noise')
parser.add_argument('--epsilon', default=0.1, type=float, help='perturbation radius')

args = parser.parse_args()

criterion = nn.CrossEntropyLoss()
# Bounds without normalization of inputs
x_min = torch.tensor([0, 0, 0]).to(device).view([1, -1, 1, 1])
x_max = torch.tensor([1, 1, 1]).to(device).view([1, -1, 1, 1])

def compute_metric(loader, net, noise_type, epsilon, max, combine, resize, dataset):
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs_pert = inputs
        if combine == True:
            corruptions = noise_type #this is just a helper
            for id, img in enumerate(inputs):
                (n, e, m) = random.choice(corruptions)
                e = float(e)
                if m == True:
                    inputs_pert[id] = sample_lp_corr(n, e, img, 'max')
                else:
                    inputs_pert[id] = sample_lp_corr(n, e, img, 'other')
        else:
            for id, img in enumerate(inputs):
                epsilon = float(epsilon)
                if max == True:
                    inputs_pert[id] = sample_lp_corr(noise_type, epsilon, img, 'max')
                else:
                    inputs_pert[id] = sample_lp_corr(noise_type, epsilon, img, 'other')
        if resize == True:
            inputs_pert = transforms.Resize(224)(inputs_pert)
        inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
        targets_pert = targets
        targets_pert_pred = net(inputs_pert)

        _, predicted = targets_pert_pred.max(1)
        total += targets_pert.size(0)
        correct += predicted.eq(targets_pert).sum().item()


    acc = 100.*correct/total
    return(acc)

def compute_metric_imagenet_c(loader_c, net):
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader_c):

        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
        inputs = inputs / 255
        targets_pred = net(inputs)

        _, predicted = targets_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return (acc)

def compute_metric_cifar_c(loader, loader_c, net, batchsize, resize):
    net.eval()
    correct = 0
    total = 0
    for intensity in range(5):
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs_pert = inputs
            for id, label in enumerate(targets):

                input_c = loader_c[intensity * 10000 + batch_idx * batchsize + id]
                inputs_pert[id] = input_c
            if resize == True:
                inputs_pert = transforms.Resize(224)(inputs)

            inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
            targets_pert = targets
            targets_pert_pred = net(inputs_pert)

            _, predicted = targets_pert_pred.max(1)
            total += targets_pert.size(0)
            correct += predicted.eq(targets_pert).sum().item()

    acc = 100. * correct / total
    return (acc)

def eval_metric(modelfilename, test_corruptions, combine_test_corruptions, test_on_c, modeltype, modelparams, resize, dataset, bsize, workers):
    if dataset == 'ImageNet':
        test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor()])
    elif dataset == 'CIFAR10' and resize == True:
        test_transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor()])
    else:
        test_transforms = transforms.Compose([transforms.ToTensor()])

    batchsize = bsize

    load_helper = getattr(datasets, dataset)
    if dataset == 'ImageNet':
        test_loader = DataLoader(torchvision.datasets.ImageFolder(root='./experiments/data/imagenet/ILSVRC/Data/val',
                                                    transform=test_transforms), batch_size=batchsize, shuffle =False,
                                 pin_memory=True, num_workers=workers)
    else:
        test_loader = DataLoader(load_helper("./experiments/data",
                            train=False, download=True, transform=test_transforms), batch_size=batchsize, shuffle =False,
                                 pin_memory=True, num_workers=workers)
    #Load model
    if modeltype == 'wrn28':
        if dataset == 'CIFAR10':
            model = WideResNet(28, 10, 0.3, 10)
        elif dataset == 'ImageNet':
            model = WideResNet(28, 10, 0.3, 1000)
    else:
        torchmodel = getattr(models, modeltype)
        model = torchmodel(**modelparams)
    model = model.to(device)

    if device == "cuda":
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    model.load_state_dict(torch.load(modelfilename)["net"])

    accs = []

    if test_on_c == True:
        corruptions = np.loadtxt('./experiments/data/labels.txt', dtype=list)
        np.asarray(corruptions)
        corruptions = np.delete(corruptions, 0) #delete the 'standard' out of the list, this is only for labeling.
        acc = compute_metric(test_loader, model, 'standard', 0.0, True, False, resize, dataset)
        accs.append(acc)
        print(acc, "% Clean Accuracy")

        if dataset == 'CIFAR10':
            print("Testing on Cifar-10-C Benchmark Noise (Hendrycks 2019)")

            for corruption in corruptions:
                test_loader_c = np.load(f'./experiments/data/cifar-10-c/{corruption}.npy')
                test_loader_c = torch.from_numpy(test_loader_c)
                test_loader_c = test_loader_c.permute(0, 3, 1, 2)
                test_loader_c = test_loader_c / 255.0
                if resize == True:
                    test_loader_c = transforms.Resize(224)(test_loader_c)
                acc = compute_metric_cifar_c(test_loader, test_loader_c, model, batchsize, resize)
                accs.append(acc)
                print(acc, "% mean (avg. over 5 intensities) Accuracy on Cifar-10-C corrupted data of type", corruption)

        elif dataset == 'ImageNet':
            print("Testing on ImageNet-C Benchmark Noise (Hendrycks 2019)")

            for corruption in corruptions:
                acc_intensities = []

                for intensity in range(1, 6):
                    load_c = datasets.ImageFolder(root='./experiments/data/imagenet-c/'+corruption+'/'+str(intensity),
                                    transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()]))
                    test_loader_c = DataLoader(load_c, batch_size=batchsize, shuffle=False)
                    acc = compute_metric_imagenet_c(test_loader_c, model)
                    acc_intensities.append(acc)
                acc = sum(acc_intensities) / 5
                accs.append(acc)
                print(acc, "% mean (avg. over 5 intensities) Accuracy on ImageNet-C corrupted data of type", corruption)

        else:
            print('No dataset corrupted benchmark other than Cifar-10-C and ImageNet-C available.')

    if combine_test_corruptions:
        print("Test Noise combined")
        acc = compute_metric(test_loader, model, test_corruptions, test_corruptions, test_corruptions,
                             combine_test_corruptions, resize, dataset)
        accs.append(acc)
    else:
        for id, (noise_type, test_epsilon, max) in enumerate(test_corruptions):
            acc = compute_metric(test_loader, model, noise_type, test_epsilon, max, combine_test_corruptions, resize, dataset)
            print(acc, "% Accuracy on random test corupptions of type:", noise_type, test_epsilon, "with maximal-perturbation =", max)
            accs.append(acc)

    return accs