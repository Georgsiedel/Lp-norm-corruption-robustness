from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import os
#os.chdir('C:\\Users\\Admin\\Desktop\\Python\\corruption-testing')
import copy
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
import gc
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
from experiments.network import WideResNet
from experiments.sample_corrupted_img import sample_lp_corr

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
                if m == 'True':
                    inputs_pert[id] = sample_lp_corr(n, e, img, 'max')
                else:
                    inputs_pert[id] = sample_lp_corr(n, e, img, 'other')
        else:
            for id, img in enumerate(inputs):
                epsilon = float(epsilon)
                if max == 'True':
                    inputs_pert[id] = sample_lp_corr(noise_type, epsilon, img, 'max')
                else:
                    inputs_pert[id] = sample_lp_corr(noise_type, epsilon, img, 'other')
        if resize == True:
            inputs_pert = transforms.Resize(224)(inputs_pert)

        if dataset == 'CIFAR10':
            inputs_pert = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))(inputs_pert)
        elif dataset == 'CIFAR100':
            inputs_pert = transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))(inputs_pert)
        elif dataset == 'ImageNet':
            inputs_pert = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(inputs_pert)

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
        inputs = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(inputs)
        targets_pred = net(inputs)

        _, predicted = targets_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return (acc)

def compute_metric_cifar10_c(loader, loader_c, net, batchsize, resize):
    net.eval()
    correct = 0
    total = 0
    for intensity in range(5):
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs_c = copy.deepcopy(inputs)
            for id, label in enumerate(targets):

                input_c = loader_c[intensity * 10000 + batch_idx * batchsize + id]
                inputs_c[id] = input_c
            if resize == True:
                inputs_c = transforms.Resize(224)(inputs_c)
            inputs_c = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))(inputs_c)

            inputs_c, targets = inputs_c.to(device, dtype=torch.float), targets.to(device)
            targets_pred = net(inputs_c)

            _, predicted = targets_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            gc.collect()

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
        testset = torchvision.datasets.ImageFolder(root='./experiments/data/imagenet/ILSVRC/Data/val', transform=test_transforms)
        test_loader = DataLoader(testset, batch_size=batchsize, shuffle =False,
                                 pin_memory=True, num_workers=workers)
    else:
        testset = load_helper("./experiments/data", train=False, download=True, transform=test_transforms)
        test_loader = DataLoader(testset, batch_size=batchsize, shuffle =False,
                                 pin_memory=True, num_workers=workers)
    num_classes = len(testset.classes)
    #Load model
    if modeltype == 'wrn28':
        if dataset == 'CIFAR10':
            model = WideResNet(28, 10, 0.3, 10)
        elif dataset == 'ImageNet':
            model = WideResNet(28, 10, 0.3, 1000)
    else:
        torchmodel = getattr(models, modeltype)
        model = torchmodel(num_classes = num_classes, **modelparams)
    model = model.to(device)

    if device == "cuda":
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    model.load_state_dict(torch.load(modelfilename)["net"])

    accs = []

    if test_on_c == True:
        corruptions = np.loadtxt('./experiments/data/c-labels.txt', dtype=list)
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
                acc = compute_metric_cifar10_c(test_loader, test_loader_c, model, batchsize, resize)
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