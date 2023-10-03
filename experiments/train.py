import argparse
import ast
import importlib
import copy
import numpy as np
from tqdm import tqdm
import shutil
import torch.nn as nn
import torch.cuda.amp
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.models as torchmodels
from sklearn.model_selection import train_test_split

from experiments.jsd_loss import JsdCrossEntropy
from experiments.data_transforms import *
import experiments.checkpoints as checkpoints
from experiments.visuals_and_reports import learning_curves
import experiments.models as low_dim_models

import torch.backends.cudnn as cudnn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.backends.cudnn.enabled = False #this may resolve some cuDNN errors, but increases training time by ~200%
torch.cuda.set_device(0)
cudnn.benchmark = True #this slightly speeds up 32bit precision training (5%)

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
parser.add_argument('--resume', type=str2bool, nargs='?', const=False, default=False,
                    help='resuming from saved checkpoint in fixed-path repo defined below')
parser.add_argument('--noise', default='standard', type=str, help='type of noise')
parser.add_argument('--epsilon', default=0.0, type=float, help='perturbation radius')
parser.add_argument('--run', default=0, type=int, help='run number')
parser.add_argument('--max', type=str2bool, nargs='?', const=False, default=False,
                    help='sample max epsilon values only (True) or random uniform values up to max epsilon (False)')
parser.add_argument('--experiment', default=0, type=int,
                    help='experiment number - each experiment is defined in module config{experiment}')
parser.add_argument('--batchsize', default=128, type=int,
                    help='Images per batch - more means quicker training, but higher memory demand')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to choose')
parser.add_argument('--validontest', type=str2bool, nargs='?', const=True, default=True, help='For datasets wihtout '
                    'standout validation (e.g. CIFAR). True: Use full training data, False: Split 20% for valiationd')
parser.add_argument('--epochs', default=100, type=int, help="number of epochs")
parser.add_argument('--learningrate', default=0.1, type=float, help='learning rate')
parser.add_argument('--lrschedule', default='MultiStepLR', type=str, help='Learning rate scheduler from pytorch.')
parser.add_argument('--lrparams', default={'milestones': [85, 95], 'gamma': 0.2}, type=str, action=str2dictAction,
                    metavar='KEY=VALUE', help='parameters for the learning rate scheduler')
parser.add_argument('--earlystop', type=str2bool, nargs='?', const=False, default=False, help='Use earlystopping after '
                    'some epochs (patience) of no increase in performance')
parser.add_argument('--earlystopPatience', default=15, type=int,
                    help='Number of epochs to wait for a better performance if earlystop is True')
parser.add_argument('--optimizer', default='SGD', type=str, help='Optimizer from torch.optim')
parser.add_argument('--optimizerparams', default={'momentum': 0.9, 'weight_decay': 5e-4}, type=str,
                    action=str2dictAction, metavar='KEY=VALUE', help='parameters for the optimizer')
parser.add_argument('--modeltype', default='wideresnet', type=str,
                    help='Modeltype to train, use either default WRN28 or model from pytorch models')
parser.add_argument('--modelparams', default={}, type=str, action=str2dictAction, metavar='KEY=VALUE',
                    help='parameters for the chosen model')
parser.add_argument('--resize', type=str2bool, nargs='?', const=False, default=False,
                    help='Resize a model to 224x224 pixels, standard for models like transformers.')
parser.add_argument('--aug_strat_check', type=str2bool, nargs='?', const=True, default=False,
                    help='Whether to use an auto-augmentation scheme')
parser.add_argument('--train_aug_strat', default='TrivialAugmentWide', type=str, help='auto-augmentation scheme')
parser.add_argument('--jsd_loss', type=str2bool, nargs='?', const=False, default=False,
                    help='Whether to use Jensen-Shannon-Divergence loss function (enforcing smoother models)')
parser.add_argument('--mixup_alpha', default=0.0, type=float, help='Mixup Alpha parameter, Pytorch suggests 0.2. If '
                    'both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance')
parser.add_argument('--cutmix_alpha', default=0.0, type=float, help='Cutmix Alpha parameter, Pytorch suggests 1.0. If '
                    'both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance')
parser.add_argument('--combine_train_corruptions', type=str2bool, nargs='?', const=True, default=True,
                    help='Whether to combine all training noise values by drawing from the randomly')
parser.add_argument('--concurrent_combinations', default=1, type=int, help='How many of the training noise values should '
                    'be applied at once on one image. USe only if you defined multiple training noise values.')
parser.add_argument('--number_workers', default=4, type=int, help='How many workers are launched to parallelize data '
                    'loading. Experimental. 4 for ImageNet, 1 for Cifar. More demand GPU memory, but maximize GPU usage.')
parser.add_argument('--lossparams', default={'num_splits': 3, 'alpha': 12, 'smoothing': 0}, type=str, action=str2dictAction, metavar='KEY=VALUE',
                    help='parameters for the JSD loss function')
parser.add_argument('--RandomEraseProbability', default=0.0, type=float,
                    help='probability of applying random erasing to an image')
parser.add_argument('--warmupepochs', default=5, type=int,
                    help='Number of Warmupepochs for stable training early on. Start with factor 10 lower learning rate')
parser.add_argument('--normalize', type=str2bool, nargs='?', const=False, default=False,
                    help='Whether to normalize input data to mean=0 and std=1')
parser.add_argument('--num_classes', default=10, type=int, help='Number of classes of the dataset')
parser.add_argument('--pixel_factor', default=1, type=int, help='default is 1 for 32px (CIFAR10), '
                    'e.g. 2 for 64px images. Scales convolutions automatically in the same model architecture')

args = parser.parse_args()
configname = (f'experiments.configs.config{args.experiment}')
config = importlib.import_module(configname)
crossentropy = nn.CrossEntropyLoss(label_smoothing=args.lossparams["smoothing"])
jsdcrossentropy = JsdCrossEntropy(**args.lossparams)

def calculate_steps(): #+0.5 is a way of rounding up to account for the last partial batch in every epoch
    if args.dataset == 'ImageNet':
        steps = round(1281167/args.batchsize + 0.5) * (args.epochs + args.warmupepochs)
        if args.validontest == True:
            steps += (round(50000/args.batchsize + 0.5) * (args.epochs + args.warmupepochs))
    if args.dataset == 'TinyImageNet':
        steps = round(100000/args.batchsize + 0.5) * (args.epochs + args.warmupepochs)
        if args.validontest == True:
            steps += (round(10000/args.batchsize + 0.5) * (args.epochs + args.warmupepochs))
    elif args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        steps = round(50000 / args.batchsize + 0.5) * (args.epochs + args.warmupepochs)
        if args.validontest == True:
            steps += (round(10000/args.batchsize + 0.5) * (args.epochs + args.warmupepochs))
    total_steps = int(steps)
    return total_steps

def train(pbar):
    model.train()
    correct, total, train_loss, avg_train_loss = 0, 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()

        inputs, targets = apply_mixing_functions(inputs, targets, args.mixup_alpha, args.cutmix_alpha, args.num_classes)

        inputs_orig, inputs_pert = copy.deepcopy(inputs), copy.deepcopy(inputs)

        if args.aug_strat_check == True:
            inputs = apply_augstrat(inputs, args.train_aug_strat)

        inputs = apply_lp_corruption(inputs, args.combine_train_corruptions, config.train_corruptions,
                                         args.concurrent_combinations, args.max, args.noise, args.epsilon)

        if args.jsd_loss == True:
            if args.aug_strat_check == True:
                inputs_pert = apply_augstrat(inputs_pert, args.train_aug_strat)
            inputs_pert = apply_lp_corruption(inputs_pert, args.combine_train_corruptions, config.train_corruptions,
                                                  args.concurrent_combinations, args.max, args.noise, args.epsilon)

        if args.jsd_loss == True:
            inputs = torch.cat((inputs_orig, inputs, inputs_pert), 0)
        if args.resize == True:
            inputs = transforms.Resize(224, antialias=True)(inputs)
        if args.normalize == True:
            inputs = normalize(inputs, args.dataset)

        inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            if args.jsd_loss == True:
                loss = jsdcrossentropy(outputs, targets)
            else:
                loss = crossentropy(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        if np.ndim(targets) == 2:
            _, targets = targets.max(1)
        if args.jsd_loss == True:
            targets = torch.cat((targets, targets, targets), 0)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        avg_train_loss = train_loss / (batch_idx + 1)
        pbar.set_description('[Train] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(
            avg_train_loss, 100. * correct / total, correct, total))
        pbar.update(1)

    train_acc = 100. * correct / total
    return train_acc, avg_train_loss

def valid(pbar):
    with torch.no_grad():
        model.eval()
        test_loss, correct, total, avg_test_loss = 0, 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(validationloader):

            if args.resize == True:
                inputs = transforms.Resize(224, antialias=True)(inputs)
            if args.normalize == True:
                inputs = normalize(inputs, args.dataset)
            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device)
            targets_pert = targets

            with torch.cuda.amp.autocast():
                targets_pert_pred = model(inputs)
                loss = crossentropy(targets_pert_pred, targets_pert)

            test_loss += loss.item()
            _, predicted = targets_pert_pred.max(1)
            total += targets_pert.size(0)
            correct += predicted.eq(targets_pert).sum().item()
            avg_test_loss = test_loss / (batch_idx + 1)
            pbar.set_description(
                '[Valid] Loss: {:.3f} | Acc: {:.3f} ({}/{})'.format(avg_test_loss, 100. * correct / total,
                                                                    correct, total))
            pbar.update(1)

        acc = 100. * correct / total
        return acc, avg_test_loss

def load_data(transform_train, transform_valid, dataset, validontest):
    if dataset == 'ImageNet' or dataset == 'TinyImageNet':
        trainset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/train',
                                                    transform=transform_train)
        if validontest == True:
            validset = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/val',
                                                        transform=transform_valid)
        else:
            trainset_clean = torchvision.datasets.ImageFolder(root=f'./experiments/data/{dataset}/train',
                                                              transform=transform_valid)
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        load_helper = getattr(torchvision.datasets, dataset)
        trainset = load_helper(root='./experiments/data', train=True, download=True, transform=transform_train)
        if validontest == True:
            validset = load_helper(root='./experiments/data', train=False, download=True, transform=transform_valid)
        else:
            trainset_clean = load_helper(root='./experiments/data', train=True, download=True,
                                         transform=transform_valid)

    if validontest == False: #validation splitting
        validsplit = 0.2
        train_indices, val_indices, _, _ = train_test_split(
            range(len(trainset)),
            trainset.targets,
            stratify=trainset.targets,
            test_size=validsplit,
            random_state=args.run)  # same validation split when calling train multiple times, but a random new validation on multiple runs
        trainset = Subset(trainset, train_indices)
        validset = Subset(trainset_clean, val_indices)

    return trainset, validset

if __name__ == '__main__':
    # Load and transform data
    print('Preparing data..')
    transform_train, transform_valid = create_transforms(args.dataset, args.RandomEraseProbability)
    trainset, validset = load_data(transform_train, transform_valid, args.dataset, args.validontest)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, pin_memory=True, collate_fn=None, num_workers=args.number_workers)
    validationloader = DataLoader(validset, batch_size=args.batchsize, shuffle=True, pin_memory=True, num_workers=args.number_workers)

    # Construct model
    print(f'\nBuilding {args.modeltype} model with {args.modelparams} | Augmentation strategy: {args.aug_strat_check}'
          f' | JSD loss: {args.jsd_loss}')
    if args.dataset == 'CIFAR10' or 'CIFAR100' or 'TinyImageNet':
        model_class = getattr(low_dim_models, args.modeltype)
        model = model_class(num_classes=args.num_classes, factor=args.pixel_factor, **args.modelparams)
    else:
        model_class = getattr(torchmodels, args.modeltype)
        model = model_class(num_classes = args.num_classes, **args.modelparams)
    model = torch.nn.DataParallel(model).to(device)

    # Define Optimizer, Learningrate Scheduler, Scaler, and Early Stopping
    opti = getattr(optim, args.optimizer)
    optimizer = opti(model.parameters(), lr=args.learningrate, **args.optimizerparams)
    schedule = getattr(optim.lr_scheduler, args.lrschedule)
    scheduler = schedule(optimizer, **args.lrparams)
    if args.warmupepochs > 0:
        warmupscheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmupepochs)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmupscheduler, scheduler], milestones=[args.warmupepochs])
    scaler = torch.cuda.amp.GradScaler()
    early_stopper = checkpoints.EarlyStopping(patience=args.earlystopPatience, verbose=False)

    # Some necessary parameters
    total_steps = calculate_steps()
    train_accs, train_losses, valid_accs, valid_losses = [], [], [], []
    training_folder = 'combined' if args.combine_train_corruptions == True else 'separate'
    filename_spec = str(f"_{args.noise}_eps_{args.epsilon}_{args.max}_" if
                        args.combine_train_corruptions == False else f"_")
    start_epoch, end_epoch = 0, args.epochs

    # Resume from checkpoint
    if args.resume == True:
        print('\nResuming from checkpoint..')
        start_epoch, model, optimizer, scheduler = checkpoints.load_model(model, optimizer, scheduler, path = 'experiments/trained_models/checkpoint.pt')

    # Training loop
    with tqdm(total=total_steps) as pbar:
        with torch.autograd.set_detect_anomaly(False, check_nan=False): #this may resolve some Cuda/cuDNN errors.
            # check_nan=True increases 32bit precision train time by ~20% and causes errors due to nan values for mixed precision training.
            for epoch in range(start_epoch, end_epoch):
                train_acc, train_loss = train(pbar)
                valid_acc, valid_loss = valid(pbar)
                train_accs.append(train_acc)
                valid_accs.append(valid_acc)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                if args.lrschedule == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()

                checkpoints.save_model(epoch, model, optimizer, scheduler, path = 'experiments/trained_models/checkpoint.pt')
                early_stopper(valid_acc, model)
                if early_stopper.best_model == True:
                    checkpoints.save_model(epoch, model, optimizer, scheduler,
                                           path='experiments/trained_models/best_checkpoint.pt')
                if args.earlystop and early_stopper.early_stop:
                    print("Early stopping")
                    end_epoch = epoch
                    break

    # Save final model
    end_epoch, model, optimizer, scheduler = checkpoints.load_model(model, optimizer, scheduler,
                                                                    path='experiments/trained_models/best_checkpoint.pt')
    checkpoints.save_model(end_epoch, model, optimizer, scheduler, path = f'./experiments/trained_models/{args.dataset}'
                                                    f'/{args.modeltype}/config{args.experiment}_{args.lrschedule}_'
                                                    f'{training_folder}{filename_spec}run_{args.run}.pth')
    # print results
    print("Maximum validation accuracy of", max(valid_accs), "achieved after", np.argmax(valid_accs) + 1, "epochs; "
         "Minimum validation loss of", min(valid_losses), "achieved after", np.argmin(valid_losses) + 1, "epochs; ")
    # save learning curves and config file
    learning_curves(args.dataset, args.modeltype, args.lrschedule, args.experiment, args.run, train_accs,
                    valid_accs, train_losses, valid_losses, training_folder, filename_spec)
    shutil.copyfile(f'./experiments/configs/config{args.experiment}.py',
                    f'./results/{args.dataset}/{args.modeltype}/config{args.experiment}_{args.lrschedule}_{training_folder}.py')