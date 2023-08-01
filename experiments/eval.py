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
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from experiments.network import WideResNet
from experiments.sample_corrupted_img import sample_lp_corr
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()
# Bounds without normalization of inputs
x_min = torch.tensor([0, 0, 0]).to(device).view([1, -1, 1, 1])
x_max = torch.tensor([1, 1, 1]).to(device).view([1, -1, 1, 1])

def compute_metric(loader, net, noise_type, epsilon, max, combine, resize, dataset, normalize):
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

        if dataset == 'CIFAR10' and normalize == True:
            inputs_pert = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))(inputs_pert)
        elif dataset == 'CIFAR100' and normalize == True:
            inputs_pert = transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))(inputs_pert)
        elif (dataset == 'ImageNet' or dataset == 'TinyImageNet') and normalize == True:
            inputs_pert = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(inputs_pert)

        inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
        targets_pert = targets
        targets_pert_pred = net(inputs_pert)

        _, predicted = targets_pert_pred.max(1)
        total += targets_pert.size(0)
        correct += predicted.eq(targets_pert).sum().item()


    acc = 100.*correct/total
    return(acc)

def compute_metric_imagenet_c(loader_c, net, normalize):
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader_c):

        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
        inputs = inputs / 255
        if normalize == True:
            inputs = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(inputs)
        targets_pred = net(inputs)

        _, predicted = targets_pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return (acc)

def compute_metric_cifar_c(loader, loader_c, net, batchsize, resize, normalize, dataset):
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
            if normalize == True and dataset == 'CIFAR10':
                inputs_c = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))(inputs_c)
            if normalize == True and dataset == 'CIFAR100':
                inputs_c = transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))(inputs_c)

            inputs_c, targets = inputs_c.to(device, dtype=torch.float), targets.to(device)
            targets_pred = net(inputs_c)

            _, predicted = targets_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            gc.collect()

    acc = 100. * correct / total
    return (acc)

def pgd_with_early_stopping(model, inputs, labels, clean_predicted, eps, number_iterations, epsilon_iters, norm):

    for i in range(number_iterations):
        adv_inputs = projected_gradient_descent(model,
                                                inputs,
                                                eps=eps,
                                                eps_iter=epsilon_iters,
                                                nb_iter=1,
                                                norm=norm,
                                                y = labels,
                                                rand_init=False,
                                                sanity_checks=False)

        adv_outputs = model(adv_inputs)
        _, adv_predicted = torch.max(adv_outputs.data, 1)

        label_flipped = bool(adv_predicted!=clean_predicted)
        if label_flipped:
            if clean_predicted!=labels:
                print(f"Iterations for successful attack on misclassified input: {i+1}")
                break
            else:
              print(f"Iterations for successful attack: {i+1}")
              break
        inputs = adv_inputs.clone()
    return adv_inputs, adv_predicted

def adv_distance(testloader, model, number_iterations, epsilon, eps_iter, norm):
    distance_list_0, image_idx_0 = [], []
    distance_list_1, image_idx_1 = [], []
    distance_list_2, image_idx_2 = [], []

    correct, total = 0, 0
    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        adv_inputs, adv_predicted = pgd_with_early_stopping(model, inputs, labels, predicted, epsilon, number_iterations, eps_iter, norm)

        distance = torch.norm((inputs - adv_inputs), p=norm)
        distance_list_0.append(distance) #all distances, also for originally misclassified points
        image_idx_0.append(i) #all points, also originally misclassified ones

        if (predicted == labels):
            distance = torch.norm((inputs - adv_inputs), p=norm)
            distance_list_1.append(distance)
            image_idx_1.append(i)
            distance_list_2.append(distance) #only originally correctly classified distances are counted
            image_idx_2.append(i) #only originally correctly classified points
        else:
            distance_list_1.append(0) #originally misclassified distances are counted as 0
            image_idx_1.append(i) #all points, also originally misclassified ones

        correct += (adv_predicted == labels).sum().item()
        total += labels.size(0)

        if i % 100 == 0:
            adv_acc = correct / total
            print(f"Completed: {i}, mean_distance: {sum(distance_list_0)/i}, Adversarial accuracy: {adv_acc * 100}%")

    return distance_list_0, image_idx_0, distance_list_1, image_idx_1, distance_list_2, image_idx_2, adv_acc

def compute_adv_distance(testset, workers, model):
    truncated_testset, _ = torch.utils.data.random_split(testset,
                                                         [1000, 9000],
                                                         generator=torch.Generator().manual_seed(42))
    truncated_testloader = DataLoader(truncated_testset, batch_size=1, shuffle=False,
                                       pin_memory=True, num_workers=workers)
    epsilon=0.1
    eps_iter=0.0004,
    nb_iters=100
    norm=np.inf

    dst0, idx0, dst1, idx1, dst2, idx2, adv_acc = adv_distance(truncated_testloader, model, nb_iters, epsilon, eps_iter, norm)

    return adv_acc*100, dst0, idx0, dst1, idx1, dst2, idx2

def eval_metric(modelfilename, test_corruptions, combine_test_corruptions, test_on_c, modeltype, modelparams, resize, dataset, bsize, workers, normalize, calculate_adv_distance):
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

    if dataset == 'ImageNet' or dataset == 'TinyImageNet':
        testset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/val', transform=test_transforms)
        test_loader = DataLoader(testset, batch_size=batchsize, shuffle =False,
                                 pin_memory=True, num_workers=workers)
    else:
        load_helper = getattr(datasets, dataset)
        testset = load_helper("./experiments/data", train=False, download=True, transform=test_transforms)
        test_loader = DataLoader(testset, batch_size=batchsize, shuffle =False,
                                 pin_memory=True, num_workers=workers)
    num_classes = len(testset.classes)
    #Load model
    if modeltype == 'wrn28':
        model = WideResNet(depth = 28, widen_factor = 10, dropout_rate=modelparams['dropout_rate'], num_classes=num_classes)
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
        acc = compute_metric(test_loader, model, 'standard', 0.0, True, False, resize, dataset, normalize)
        accs.append(acc)
        print(acc, "% Clean Accuracy")
        print(f"Testing on {dataset}-c Benchmark Noise (Hendrycks 2019)")
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            for corruption in corruptions:
                test_loader_c = np.load(f'./experiments/data/{dataset}-c/{corruption}.npy')
                test_loader_c = torch.from_numpy(test_loader_c)
                test_loader_c = test_loader_c.permute(0, 3, 1, 2)
                test_loader_c = test_loader_c / 255.0
                if resize == True:
                    test_loader_c = transforms.Resize(224)(test_loader_c)
                acc = compute_metric_cifar_c(test_loader, test_loader_c, model, batchsize, resize, normalize, dataset)
                accs.append(acc)
                print(acc, f"% mean (avg. over 5 intensities) Accuracy on {dataset}-c corrupted data of type", corruption)

        elif dataset == 'ImageNet' or dataset == 'TinyImageNet':
            for corruption in corruptions:
                acc_intensities = []

                for intensity in range(1, 6):
                    load_c = datasets.ImageFolder(root=f'./experiments/data/{dataset}-c/'+corruption+'/'+str(intensity),
                                    transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()]))
                    test_loader_c = DataLoader(load_c, batch_size=batchsize, shuffle=False)
                    acc = compute_metric_imagenet_c(test_loader_c, model, normalize)
                    acc_intensities.append(acc)
                acc = sum(acc_intensities) / 5
                accs.append(acc)
                print(acc, f"% mean (avg. over 5 intensities) Accuracy on {dataset}-c corrupted data of type", corruption)

        else:
            print('No corrupted benchmark available other than CIFAR10-c, CIFAR100-c, TinyImageNet-c and ImageNet-c.')

    if combine_test_corruptions:
        print("Test Noise combined")
        acc = compute_metric(test_loader, model, test_corruptions, test_corruptions, test_corruptions,
                             combine_test_corruptions, resize, dataset, normalize)
        accs.append(acc)
    else:
        for id, (noise_type, test_epsilon, max) in enumerate(test_corruptions):
            acc = compute_metric(test_loader, model, noise_type, test_epsilon, max, combine_test_corruptions, resize, dataset, normalize)
            print(acc, "% Accuracy on random test corupptions of type:", noise_type, test_epsilon, "with maximal-perturbation =", max)
            accs.append(acc)

    if calculate_adv_distance == True:
        adv_acc_high_iter_pgd, dst0, idx0, dst1, idx1, dst2, idx2 = compute_adv_distance(testset, workers, model)
        accs.append(adv_acc_high_iter_pgd)
        distances = [torch.tensor(d).cpu().np() for d in [dst0, dst1, dst2]]
        mean_dist0, mean_dist1, mean_dist2 = [dist.mean() for dist in distances]
        accs.append([mean_dist0, mean_dist1, mean_dist2])

    return accs