import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchmetrics.classification import MulticlassCalibrationError
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.device(0)

import experiments.models.smallsized as smallsized
import experiments.models.ImageNet as ImageNet
from experiments.data_transforms import apply_lp_corruption
from experiments.normalized_model_wrapper import create_normalized_model_wrapper
import experiments.adversarial_eval as adv_eval

def compute_metric(loader, net, noise_type, combine, resize):
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)

            inputs_pert = apply_lp_corruption(inputs, minibatchsize=8, combine_train_corruptions=combine, corruptions=noise_type,
                                              concurrent_combinations=1, noise_patch_scale=1.0, random_noise_dist=None)

            if resize == True:
                inputs_pert = transforms.Resize(224, antialias=True)(inputs_pert)

            targets_pert = targets
            with torch.cuda.amp.autocast():
                targets_pert_pred = net(inputs_pert)

            _, predicted = targets_pert_pred.max(1)
            total += targets_pert.size(0)
            correct += predicted.eq(targets_pert).sum().item()

        acc = 100.*correct/total
        return(acc)

def compute_clean(loader, net, resize, num_classes):
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        calibration_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=100, norm='l2')
        all_targets = torch.empty(0)
        all_targets_pred = torch.empty((0, num_classes))
        all_targets, all_targets_pred = all_targets.to(device), all_targets_pred.to(device)

        for batch_idx, (inputs, targets) in enumerate(loader):
            if resize == True:
                inputs = transforms.Resize(224, antialias=True)(inputs)

            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            with torch.cuda.amp.autocast():
                targets_pred = net(inputs)

            _, predicted = targets_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_targets = torch.cat((all_targets, targets), 0)
            all_targets_pred = torch.cat((all_targets_pred, targets_pred), 0)

        acc = 100.*correct/total
        rmsce_clean = float(calibration_metric(all_targets_pred, all_targets).cpu())

        return acc, rmsce_clean

def compute_metric_imagenet_c(loader_c, net, num_classes):
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        calibration_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=100, norm='l2')
        all_targets = torch.empty(0)
        all_targets_pred = torch.empty((0, num_classes))
        all_targets, all_targets_pred = all_targets.to(device), all_targets_pred.to(device)

        for batch_idx, (inputs, targets) in enumerate(loader_c):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            with torch.cuda.amp.autocast():
                targets_pred = net(inputs)

            _, predicted = targets_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_targets = torch.cat((all_targets, targets), 0)
            all_targets_pred = torch.cat((all_targets_pred, targets_pred), 0)

        rmsce_c = float(calibration_metric(all_targets_pred, all_targets).cpu())
        acc = 100. * correct / total

        return acc, rmsce_c

def compute_metric_cifar_c(loader, loader_c, net, batchsize, num_classes):
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        calibration_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=100, norm='l2')
        all_targets = torch.empty(0)
        all_targets_pred = torch.empty((0, num_classes))
        all_targets, all_targets_pred = all_targets.to(device), all_targets_pred.to(device)

        for intensity in range(5):
            for batch_idx, (inputs, targets) in enumerate(loader):
                for id, label in enumerate(targets):

                    input_c = loader_c[intensity * 10000 + batch_idx * batchsize + id]
                    inputs[id] = input_c

                inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
                with torch.cuda.amp.autocast():
                    targets_pred = net(inputs)

                _, predicted = targets_pred.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_targets = torch.cat((all_targets, targets), 0)
                all_targets_pred = torch.cat((all_targets_pred, targets_pred), 0)

        acc = 100.*correct/total
        rmsce_clean = float(calibration_metric(all_targets_pred, all_targets).cpu())

        return acc, rmsce_clean

def eval_metric(modelfilename, test_corruptions, combine_test_corruptions, test_on_c, modeltype, modelparams, resize,
                dataset, batchsize, workers, normalize, calculate_adv_distance, adv_distance_params,
                calculate_autoattack_robustness, autoattack_params, pixel_factor):
    if dataset == 'ImageNet':
        test_transforms = transforms.Compose([transforms.Resize(256, antialias=True),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor()])
    elif resize == True:
        test_transforms = transforms.Compose([transforms.Resize(224, antialias=True),
                                              transforms.ToTensor()])
    else:
        test_transforms = transforms.Compose([transforms.ToTensor()])

    if dataset == 'ImageNet' or dataset == 'TinyImageNet':
        testset = torchvision.datasets.ImageFolder(root=f'../corruption-testing/experiments/data/{dataset}/val', transform=test_transforms)
        test_loader = DataLoader(testset, batch_size=batchsize, shuffle =False,
                                 pin_memory=True, num_workers=workers)
    else:
        load_helper = getattr(datasets, dataset)
        testset = load_helper("./experiments/data", train=False, download=True, transform=test_transforms)
        test_loader = DataLoader(testset, batch_size=batchsize, shuffle =False,
                                 pin_memory=True, num_workers=workers)
    num_classes = len(testset.classes)

    #Load model
    if dataset in ('CIFAR10', 'CIFAR100', 'TinyImageNet'):
        model_class = getattr(smallsized, modeltype)
        model = model_class(num_classes=num_classes, factor=pixel_factor, **modelparams)
        if normalize == True:
            Normalized_Model_Wrapper = create_normalized_model_wrapper(dataset, modeltype)
            model = Normalized_Model_Wrapper(num_classes=num_classes, factor=pixel_factor, **modelparams)
    else:
        model_class = getattr(ImageNet, modeltype)
        model = model_class(num_classes = num_classes, **modelparams)
        if normalize == True:
            Normalized_Model_Wrapper = create_normalized_model_wrapper(dataset, modeltype)
            model = Normalized_Model_Wrapper(num_classes=num_classes, **modelparams)
    model = torch.nn.DataParallel(model).to(device)
    cudnn.benchmark = True
    model.load_state_dict(torch.load(modelfilename)["model_state_dict"], strict=False)

    accs = []
    acc, rmsce = compute_clean(test_loader, model, resize, num_classes)
    accs = accs + [acc, rmsce]
    print("Clean Accuracy ",acc,"%, RMSCE Calibration Error: ", rmsce)

    if test_on_c == True:
        corruptions = np.loadtxt('./experiments/data/c-labels.txt', dtype=list)
        np.asarray(corruptions)
        rmsce_c_list = []
        print(f"Testing on {dataset}-c Benchmark Noise (Hendrycks 2019)")
        if dataset == 'CIFAR10' or dataset == 'CIFAR100':
            for corruption in corruptions:
                test_loader_c = np.load(f'./experiments/data/{dataset}-c/{corruption}.npy')
                test_loader_c = torch.from_numpy(test_loader_c)
                test_loader_c = test_loader_c.permute(0, 3, 1, 2)
                test_loader_c = test_loader_c / 255.0
                if resize == True:
                    test_loader_c = transforms.Resize(224, antialias=True)(test_loader_c)
                acc, rmsce_c = compute_metric_cifar_c(test_loader, test_loader_c, model, batchsize, num_classes)
                accs.append(acc)
                rmsce_c_list.append(rmsce_c)
                print(acc, f"% mean (avg. over 5 intensities) Accuracy on {dataset}-c corrupted data of type", corruption)

        elif dataset == 'ImageNet' or dataset == 'TinyImageNet':
            for corruption in corruptions:
                acc_intensities = []

                for intensity in range(1, 6):
                    load_c = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}-c/'+corruption+'/'+str(intensity),
                                    transform=test_transforms)
                    test_loader_c = DataLoader(load_c, batch_size=batchsize, shuffle=False)
                    acc, rmsce_c = compute_metric_imagenet_c(test_loader_c, model, num_classes)
                    acc_intensities.append(acc)
                    rmsce_c_list.append(rmsce_c)
                acc = sum(acc_intensities) / 5
                accs.append(acc)
                print(acc, f"% mean (avg. over 5 intensities) Accuracy on {dataset}-c corrupted data of type", corruption)

        else:
            print('No corrupted benchmark available other than CIFAR10-c, CIFAR100-c, TinyImageNet-c and ImageNet-c.')

        rmsce_c = np.average(np.asarray(rmsce_c_list))
        print("Average Robust Accuracy (all 19 corruptions): ",sum(accs[2:21])/19,"%, Average Robust Accuracy (15 corruptions): ",sum(accs[2:17])/15,"%, RMSCE-C: ", rmsce_c)
        accs.append(sum(accs[2:21])/19)
        accs.append(sum(accs[2:17])/15)
        accs.append(rmsce_c)

    if calculate_adv_distance == True:
        print(f"{adv_distance_params['norm']}-Adversarial Distance calculation using PGD attack with {adv_distance_params['nb_iters']} iterations of "
              f"stepsize {adv_distance_params['eps_iter']}")
        adv_acc_high_iter_pgd, dst1, idx1, dst2, idx2 = adv_eval.compute_adv_distance(testset, workers, model, adv_distance_params)
        accs.append(adv_acc_high_iter_pgd)
        mean_dist1, mean_dist2 = [np.asarray(torch.tensor(d).cpu()).mean() for d in [dst1, dst2]]
        accs = accs + [mean_dist1, mean_dist2]
    if calculate_autoattack_robustness == True:
        print(f"{autoattack_params['norm']} Adversarial Accuracy calculation using AutoAttack attack with epsilon={autoattack_params['epsilon']}")

        adv_acc_aa, mean_dist_aa = adv_eval.compute_adv_acc(autoattack_params, testset, model, workers, batchsize)
        accs = accs + [adv_acc_aa, mean_dist_aa]
    if combine_test_corruptions:
        acc = compute_metric(test_loader, model, test_corruptions, True, resize)
        print(acc, "% Accuracy on combined Lp-norm Test Noise")
        accs.append(acc)
    else:
        for id, (testcorruption) in enumerate(test_corruptions):
            acc = compute_metric(test_loader, model, testcorruption, False, resize)
            print(acc, "% Accuracy on random test coruptions of type:", testcorruption)
            accs.append(acc)

    return accs