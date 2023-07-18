from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import random
import importlib

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
# os.chdir('C:\\Users\\Admin\\Desktop\\Python\\corruption-testing')

import torchmetrics.classification
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
import torch.cuda.amp
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms.autoaugment import AugMix
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate
from timm.loss import JsdCrossEntropy
from experiments.augmix_orig import AugMixDataset
from experiments.network import WideResNet
from experiments.sample_corrupted_img import sample_lp_corr
from experiments.earlystopping import EarlyStopping
import experiments.mix_transforms as mix_transforms

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(0)
torch.backends.cudnn.enabled = False
cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Error: Boolean value expected for argument {v}.')


class str2dictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Parse the dictionary string into a dictionary object
        # if values == '':

        try:
            dictionary = ast.literal_eval(values)
            if not isinstance(dictionary, dict):
                raise ValueError("Invalid dictionary format")
        except (ValueError, SyntaxError) as e:
            raise argparse.ArgumentTypeError(f"Invalid dictionary format: {values}") from e

        setattr(namespace, self.dest, dictionary)


parser = argparse.ArgumentParser(description='PyTorch Training with perturbations')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noise', default='gaussian', type=str, help='type of noise')
parser.add_argument('--epsilon', default=0.0, type=float, help='perturbation radius')
parser.add_argument('--run', default=0, type=int, help='run number')
parser.add_argument('--max', type=str2bool, nargs='?', const=False, default=False,
                    help='sample max epsilon values only (True) or random uniform values up to max epsilon (False)')
parser.add_argument('--experiment', default=0, type=int,
                    help='experiment number - each experiment is defined in module config{experiment}')
parser.add_argument('--batchsize', default=128, type=int,
                    help='Images per batch - more means quicker training, but higher memory demand')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to choose')
parser.add_argument('--validontest', type=str2bool, nargs='?', const=True, default=True,
                    help='For datasets wihtout standout validation (e.g. CIFAR). True: Use full training data, False: Split 20% for valiationd')
parser.add_argument('--epochs', default=100, type=int, help="number of epochs")
parser.add_argument('--learningrate', default=0.1, type=float, help='learning rate')
parser.add_argument('--lrschedule', default='MultiStepLR', type=str, help='Learning rate scheduler from pytorch.')
parser.add_argument('--lrparams', default={'milestones': [85, 95], 'gamma': 0.2}, type=str, action=str2dictAction,
                    metavar='KEY=VALUE', help='parameters for the learning rate scheduler')
parser.add_argument('--earlystop', type=str2bool, nargs='?', const=False, default=False,
                    help='Use earlystopping after some epochs (patience) of no increase in performance, then break training and reset to best checkpoint')
parser.add_argument('--earlystopPatience', default=15, type=int,
                    help='Number of epochs to wait for a better performance if earlystop is True')
parser.add_argument('--optimizer', default='SGD', type=str, help='Optimizer from torch.optim')
parser.add_argument('--optimizerparams', default={'momentum': 0.9, 'weight_decay': 5e-4}, type=str,
                    action=str2dictAction, metavar='KEY=VALUE', help='parameters for the optimizer')
parser.add_argument('--modeltype', default='wrn28', type=str,
                    help='Modeltype to train, use either defualt WRN28 or model from pytorch models')
parser.add_argument('--modelparams', default="{}", type=str, action=str2dictAction, metavar='KEY=VALUE',
                    help='parameters for the chosen model')
parser.add_argument('--resize', type=str2bool, nargs='?', const=False, default=False,
                    help='Resize a model to 224x224 pixels, standard for models like transformers.')
parser.add_argument('--aug_strat_check', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to use an auto-augmentation scheme')
parser.add_argument('--train_aug_strat', default='TrivialAugmentWide', type=str, help='auto-augmentation scheme')
parser.add_argument('--jsd_loss', type=str2bool, nargs='?', const=False, default=False,
                    help='Whether to use Jensen-Shannon-Divergence loss function (enforcing smoother models)')
parser.add_argument('--mixup_alpha', default=0.0, type=float,
                    help='Mixup Alpha parameter, Pytorch suggests 0.2. If both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance')
parser.add_argument('--cutmix_alpha', default=0.0, type=float,
                    help='Cutmix Alpha parameter, Pytorch suggests 1.0. If both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance')
parser.add_argument('--combine_train_corruptions', type=str2bool, nargs='?', const=True, default=True,
                    help='Whether to combine all training noise values by drawing from the randomly')
parser.add_argument('--concurrent_combinations', default=1, type=int,
                    help='How many of the training noise values should be applied at once on one image. USe only if you defined multiple training noise values.')

args = parser.parse_args()

configname = (f'experiments.configs.config{args.experiment}')
config = importlib.import_module(configname)

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
crossentropy = nn.CrossEntropyLoss()
jsdcrossentropy = JsdCrossEntropy(num_splits=3, alpha=12, smoothing=0)
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

        if args.aug_strat_check == True and args.train_aug_strat == 'AugMix' and args.jsd_loss == True:  # split the three splits given in the batch through the AugMix function called in main
            inputs = torch.cat(inputs, 0)
            inputs_orig, inputs, inputs_pert = torch.split(inputs, [batchsize_train, batchsize_train, batchsize_train])
        else:  # create 3 splits for JSD loss
            inputs_orig = inputs
            inputs_pert = inputs

        if args.aug_strat_check == True:
            if not args.train_aug_strat == 'AugMix':
                inputs = inputs * 255.0
                inputs = torch.clip(inputs, 0.0, 255.0)
                inputs = inputs.type(torch.uint8)
                tf = getattr(transforms, args.train_aug_strat)
                inputs = tf()(inputs)
                inputs = inputs.type(torch.float32) / 255.0
                if args.jsd_loss == True:
                    inputs_pert = inputs_pert * 255.0
                    inputs_pert = torch.clip(inputs_pert, 0.0, 255.0)
                    inputs_pert = inputs_pert.type(torch.uint8)
                    tf = getattr(transforms, args.train_aug_strat)
                    inputs_pert = tf()(inputs_pert)
                    inputs_pert = inputs_pert.type(torch.float32) / 255.0
            elif args.train_aug_strat == 'AugMix' and args.jsd_loss == False:
                inputs = inputs * 255.0
                inputs = torch.clip(inputs, 0.0, 255.0)
                inputs = inputs.type(torch.uint8)
                inputs = AugMix()(inputs)
                inputs = inputs.type(torch.float32) / 255.0
                inputs_pert = inputs

        if args.combine_train_corruptions == True:
            for id, img in enumerate(inputs_pert):
                corruptions_list = random.sample(list(config.train_corruptions), k=args.concurrent_combinations)
                for x, (noise_type, train_epsilon, max) in enumerate(corruptions_list):
                    train_epsilon = float(train_epsilon)
                    if max == False:
                        img = sample_lp_corr(noise_type, train_epsilon, img, 'other')
                    elif max == True:
                        img = sample_lp_corr(noise_type, train_epsilon, img, 'max')
                inputs_pert[id] = img
        else:
            for id, img in enumerate(inputs_pert):
                if args.max == False:
                    inputs_pert[id] = sample_lp_corr(args.noise, args.epsilon, img, 'other')
                elif args.max == True:
                    inputs_pert[id] = sample_lp_corr(args.noise, args.epsilon, img, 'max')

        if args.jsd_loss == True:
            inputs_pert = torch.cat((inputs_orig, inputs, inputs_pert), 0)

        if args.resize == True and args.dataset == 'CIFAR10':
            inputs_pert = transforms.Resize(224)(inputs_pert)
        inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
        with torch.cuda.amp.autocast():
            outputs = net(inputs_pert)
            if args.jsd_loss == True:
                loss = jsdcrossentropy(outputs, targets)
            else:
                loss = crossentropy(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        if np.ndim(targets) == 2:
            _, targets = targets.max(1)
        if args.jsd_loss == True:
            targets = torch.cat((targets, targets, targets), 0)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description('[Train] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(train_loss / (batch_idx + 1),
                                                                                 100. * correct / total,
                                                                                 correct, total))
        pbar.update(1)

    train_acc = 100. * correct / total
    return train_acc


def valid(pbar):
    """ Test current network on validation set"""
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(validationloader):
        inputs_pert = inputs

        if args.resize == True and args.dataset == 'CIFAR10':
            inputs_pert = transforms.Resize(224)(inputs_pert)

        inputs_pert, targets = inputs_pert.to(device, dtype=torch.float), targets.to(device)
        targets_pert = targets
        targets_pert_pred = net(inputs_pert)

        loss = crossentropy(targets_pert_pred, targets_pert)
        test_loss += loss.item()
        _, predicted = targets_pert_pred.max(1)
        total += targets_pert.size(0)
        correct += predicted.eq(targets_pert).sum().item()

        pbar.set_description(
            '[Valid] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(test_loss / (batch_idx + 1), 100. * correct / total,
                                                                correct, total))
        pbar.update(1)

    acc = 100. * correct / total
    return acc, test_loss


if __name__ == '__main__':
    # Load and transform data
    print('Preparing data..')

    # list of all data transformations used
    t = transforms.ToTensor()
    c32 = transforms.RandomCrop(32, padding=4)
    flip = transforms.RandomHorizontalFlip()
    r256 = transforms.Resize(256)
    c224 = transforms.CenterCrop(224)
    rrc224 = transforms.RandomResizedCrop(224)
    # transformations of validation set
    if args.dataset == 'CIFAR10':
        transform_valid = transforms.Compose([t])
    elif args.dataset == 'ImageNet':
        transform_valid = transforms.Compose([t, r256, c224])
    # transformations of training set

    transform_train = transforms.Compose([flip])
    if args.dataset == 'CIFAR10':  # and args.aug_strat_check == True and args.train_aug_strat == 'AugMix':
        transform_train = transforms.Compose([transform_train, c32])
    elif args.dataset == 'ImageNet':  # and args.aug_strat_check == True and args.train_aug_strat == 'AugMix':
        transform_train = transforms.Compose([transform_train, rrc224])
    if not (args.aug_strat_check == True and args.train_aug_strat == 'AugMix' and args.jsd_loss == True):
        transform_train = transforms.Compose([transform_train, t])

    load_helper = getattr(torchvision.datasets, args.dataset)

    if args.dataset == 'ImageNet':
        trainset = torchvision.datasets.ImageFolder(root='./experiments/data/imagenet/ILSVRC/Data/train',
                                                    transform=transform_train)
        if args.validontest == True:
            validset = torchvision.datasets.ImageFolder(root='./experiments/data/imagenet/ILSVRC/Data/val',
                                                        transform=transform_valid)
        else:
            trainset_clean = torchvision.datasets.ImageFolder(root='./experiments/data/imagenet/ILSVRC/Data/train',
                                                              transform=transform_valid)

    if args.dataset == 'CIFAR10':
        trainset = load_helper(root='./experiments/data', train=True, download=True, transform=transform_train)
        if args.validontest == True:
            validset = load_helper(root='./experiments/data', train=False, download=True, transform=transform_valid)
        else:
            trainset_clean = load_helper(root='./experiments/data', train=True, download=True,
                                         transform=transform_valid)

    if args.validontest == False:
        train_indices, val_indices, _, _ = train_test_split(
            range(len(trainset)),
            trainset.targets,
            stratify=trainset.targets,
            test_size=0.2,
            random_state=args.run)  # same validation split when calling train multiple times, but a random new validation on multiple runs
        # generate subset based on indices
        trainset = Subset(trainset, train_indices)
        validset = Subset(trainset_clean, val_indices)

    num_classes = len(trainset.classes)

    if args.aug_strat_check == True and args.train_aug_strat == 'AugMix' and args.jsd_loss == True:
        img_dim = 32 if args.dataset == 'CIFAR10' else 224
        trainset = AugMixDataset(trainset, preprocess=transforms.Compose([transforms.ToTensor()]), width=3, severity=3,
                                 img_size=img_dim)

    collate_fn = None
    mixes = []
    if args.mixup_alpha > 0.0:
        mixes.append(mix_transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
        print('Mixup is used for Training')
    if args.cutmix_alpha > 0.0:
        mixes.append(mix_transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
        print('Cutmix is used for Training')
    if mixes:
        mixupcutmix = torchvision.transforms.RandomChoice(mixes)


        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    # create batches
    batchsize_train = args.batchsize
    batchsize_valid = args.batchsize

    trainloader = DataLoader(trainset, batch_size=batchsize_train, shuffle=True, pin_memory=True, collate_fn=collate_fn, num_workers=0)
    validationloader = DataLoader(validset, batch_size=batchsize_valid, shuffle=True, pin_memory=True, num_workers=0)

    # Construct model
    print('\nBuilding', args.modeltype, 'model')
    if args.modeltype == 'wrn28':
        net = WideResNet(28, 10, 0.3, num_classes)
    else:
        torchmodel = getattr(models, args.modeltype)
        net = torchmodel(**args.modelparams)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    if args.resume:
        # Load checkpoint.
        print('\nResuming from checkpoint..')
        if not args.combine_train_corruptions:
            checkpoint = torch.load(f'./experiments/models/{args.dataset}/{args.modeltype}/{args.lrschedule}/'
                                    f'separate_training/{args.modeltype}_{args.noise}_epsilon_{args.epsilon}_'
                                    f'{args.max}_run_{args.run}.pth')
        else:
            checkpoint = torch.load(f'./experiments/models/{args.dataset}/{args.modeltype}/{args.lrschedule}/'
                                    f'combined_training/{args.modeltype}_config{args.experiment}_concurrent_'
                                    f'{args.concurrent_combinations}_run_{args.run}.pth')

        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1

    # Number of batches
    if args.dataset == 'ImageNet':
        total_steps = (1281167 / batchsize_train + 50000 / batchsize_valid) * args.epochs
    elif args.dataset == 'CIFAR10' and args.validontest == True:
        total_steps = (50000 / batchsize_train + 10000 / batchsize_valid) * args.epochs
    elif args.dataset == 'CIFAR10' and args.validontest == False:
        total_steps = (40000 / batchsize_train + 10000 / batchsize_valid) * args.epochs

    opti = getattr(optim, args.optimizer)
    optimizer = opti(net.parameters(), lr=args.learningrate, **args.optimizerparams)

    schedule = getattr(optim.lr_scheduler, args.lrschedule)
    scheduler = schedule(optimizer, **args.lrparams)
    scaler = torch.cuda.amp.GradScaler()

    early_stopping = EarlyStopping(patience=args.earlystopPatience, verbose=False, path='experiments/checkpoint.pt')
    train_accs = []
    valid_accs = []
    # Training loop
    print('\nTraining model..')
    if args.aug_strat_check == True:
        print(args.train_aug_strat, 'is used')
    if args.jsd_loss == True:
        print('JSD loss is used')
    with tqdm(total=total_steps) as pbar:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            train_acc = train(pbar)
            valid_acc, valid_loss = valid(pbar)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)

            if args.lrschedule == 'ReduceLROnPlateau':
                scheduler.step(valid_loss)
            else:
                scheduler.step()

            early_stopping(valid_loss, net)
            if args.earlystop and early_stopping.early_stop:
                print("Early stopping")
                break

        # Save best epoch
        net.load_state_dict(torch.load('experiments/checkpoint.pt'))
        state = {
            'net': net.state_dict(),
            # 'acc': max(valid_accs),
            # 'train_acc': train_accs[np.argmax(valid_accs)],
            # 'epoch': start_epoch-1+np.argmax(valid_accs)+1,
        }
        if args.combine_train_corruptions == True:
            torch.save(state,
                       f'./experiments/models/{args.dataset}/{args.modeltype}/{args.lrschedule}/combined_training/'
                       f'{args.modeltype}_config{args.experiment}_concurrent_{args.concurrent_combinations}_run_'
                       f'{args.run}.pth')
        else:
            torch.save(state,
                       f'./experiments/models/{args.dataset}/{args.modeltype}/{args.lrschedule}/separate_training/'
                       f'{args.modeltype}_{args.noise}_epsilon_{args.epsilon}_{args.max}_run_{args.run}.pth')

        print("Maximum validation accuracy of", max(valid_accs), "achieved after", np.argmax(valid_accs) + 1, "epochs")

        if args.combine_train_corruptions:
            if args.resume:
                old_train_accs = np.loadtxt(
                    f'results/{args.dataset}/{args.modeltype}/{args.lrschedule}/combined_training/'
                    f'{args.modeltype}_config{args.experiment}_learning_curve_train_run'
                    f'_{args.run}.csv', delimiter=';')
                train_accs = np.append(old_train_accs, train_accs)
                old_valid_accs = np.loadtxt(
                    f'results/{args.dataset}/{args.modeltype}/{args.lrschedule}/combined_training/'
                    f'{args.modeltype}_config{args.experiment}_learning_curve_valid_run_'
                    f'{args.run}.csv', delimiter=';')
                valid_accs = np.append(old_valid_accs, valid_accs)
            np.savetxt(f'results/{args.dataset}/{args.modeltype}/{args.lrschedule}/combined_training/{args.modeltype}_'
                       f'config{args.experiment}_learning_curve_train_run_{args.run}.csv', train_accs, delimiter=';')
            np.savetxt(f'results/{args.dataset}/{args.modeltype}/{args.lrschedule}/combined_training/{args.modeltype}_'
                       f'config{args.experiment}_learning_curve_valid_run_{args.run}.csv', valid_accs, delimiter=';')
        else:
            if args.resume:
                old_train_accs = np.loadtxt(
                    f'results/{args.dataset}/{args.modeltype}/{args.lrschedule}/separate_training/'
                    f'{args.modeltype}_config{args.experiment}_learning_curve_train_'
                    f'{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.csv', delimiter=';')
                train_accs = np.append(old_train_accs, train_accs)
                old_valid_accs = np.loadtxt(
                    f'results/{args.dataset}/{args.modeltype}/{args.lrschedule}/separate_training/'
                    f'{args.modeltype}_config{args.experiment}_learning_curve_valid_'
                    f'{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.csv', delimiter=';')
                valid_accs = np.append(old_valid_accs, valid_accs)
            np.savetxt(f'results/{args.dataset}/{args.modeltype}/{args.lrschedule}/separate_training/{args.modeltype}_'
                       f'config{args.experiment}_learning_curve_train_{args.noise}_{args.epsilon}_{args.max}_run_'
                       f'{args.run}.csv', train_accs, delimiter=';')
            np.savetxt(f'results/{args.dataset}/{args.modeltype}/{args.lrschedule}/separate_training/{args.modeltype}_'
                       f'config{args.experiment}_learning_curve_valid_{args.noise}_{args.epsilon}_{args.max}_run_'
                       f'{args.run}.csv', valid_accs, delimiter=';')

        x = list(range(1, len(train_accs) + 1))
        plt.plot(x, train_accs, label='Train Accuracy')
        plt.plot(x, valid_accs, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.xticks(np.linspace(1, len(train_accs), num=10, dtype=int))
        plt.legend(loc='best')
        if args.combine_train_corruptions:
            plt.savefig(f'results/{args.dataset}/{args.modeltype}/{args.lrschedule}/combined_training/{args.modeltype}_'
                        f'config{args.experiment}_learning_curve_run_{args.run}.svg')
        else:
            plt.savefig(f'results/{args.dataset}/{args.modeltype}/{args.lrschedule}/separate_training/{args.modeltype}_'
                        f'config{args.experiment}_learning_curve_{args.noise}_{args.epsilon}_{args.max}_run_{args.run}.svg')