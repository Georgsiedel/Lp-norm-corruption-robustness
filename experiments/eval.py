from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

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
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import os
from experiments.network import WideResNet
from experiments.sample_corrupted_img import sample_lp_corr

criterion = nn.CrossEntropyLoss()
# Bounds without normalization of inputs
x_min = torch.tensor([0, 0, 0]).to(device).view([1, -1, 1, 1])
x_max = torch.tensor([1, 1, 1]).to(device).view([1, -1, 1, 1])

def compute_metric(loader, net, noise_type, epsilon, max, combine):
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        #print("Batch", batch_idx, "out of 200 at", time.perf_counter())
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

        inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
        targets_pert = targets
        targets_pert_pred = net(inputs_pert)

        _, predicted = targets_pert_pred.max(1)
        total += targets_pert.size(0)
        correct += predicted.eq(targets_pert).sum().item()


    acc = 100.*correct/total
    return(acc)


def compute_metric_c(loader, loader_c, net, batchsize):
    net.eval()
    correct = 0
    total = 0
    for intensity in range(5):
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs_pert = inputs
            for id, label in enumerate(targets):

                input_c = loader_c[intensity * 10000 + batch_idx * batchsize + id]
                inputs_pert[id] = input_c
            inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
            targets_pert = targets
            targets_pert_pred = net(inputs_pert)

            _, predicted = targets_pert_pred.max(1)
            total += targets_pert.size(0)
            correct += predicted.eq(targets_pert).sum().item()

    acc = 100. * correct / total
    return (acc)

def eval_metric(modelfilename, test_corruptions, combine_test_corruptions, test_on_c, modeltype, modelspecs, bsize):
    test_transforms=transforms.Compose([transforms.ToTensor()])
    batchsize = bsize
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./experiments/data', train=False, download=True, transform=test_transforms),
        batch_size=batchsize, shuffle=False)
    # Load model
    if modeltype == 'wrn28':
        model = WideResNet(28, 10, 0.3, 10)
    else:
        torchmodel = getattr(models, modeltype)
        model = torchmodel(**modelspecs)
    model = model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    model.load_state_dict(torch.load(modelfilename)["net"])

    accs = []
    if test_on_c == True:
        print("Testing on Cifar-C Benchmark Noise (Hendrycks 2019)")
        corruptions = np.loadtxt('./experiments/data/cifar-10-c/labels.txt', dtype=list)
        np.asarray(corruptions)
        corruptions = np.delete(corruptions, 0) #delete the 'standard' out of the list, this is only for labeling.
        acc = compute_metric(test_loader, model, 'standard', 0.0, True, False)
        accs.append(acc)
        print(acc, "% Clean Accuracy")
        for corruption in corruptions:
            test_loader_c = np.load(f'./experiments/data/cifar-10-c/{corruption}.npy')
            test_loader_c = torch.from_numpy(test_loader_c)
            test_loader_c = test_loader_c.permute(0, 3, 1, 2)
            test_loader_c = test_loader_c / 255.0
            acc = compute_metric_c(test_loader, test_loader_c, model, batchsize)
            accs.append(acc)
            print(acc, "% mean (avg. over 5 intensities) Accuracy on Cifar-C corrupted data of type", corruption)
    if combine_test_corruptions == True:
        print("Test Noise combined")
        acc = compute_metric(test_loader, model, test_corruptions, test_corruptions, test_corruptions,
                             combine_test_corruptions)
        accs.append(acc)
    if combine_test_corruptions == False:
        for id, (noise_type, test_epsilon, max) in enumerate(test_corruptions):
            acc = compute_metric(test_loader, model, noise_type, test_epsilon, max, combine_test_corruptions)
            print(acc, "% Accuracy on random test corupptions of type:", noise_type, test_epsilon, "with maximal-perturbation = ", max)
            accs.append(acc)
    return accs