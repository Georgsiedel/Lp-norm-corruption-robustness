"""
This code is based on the following two repositories
* https://github.com/google-research/augmix
* https://github.com/erichson/NoisyMixup
"""

import argparse
import os

import augmentations
import numpy as np

from src.cifar_models import preactwideresnet18, preactresnet18, wideresnet28

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from src.noisy_mixup import mixup_criterion
from src.tools import get_lr
from aug_utils import *

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--arch', '-m', type=str, default='wideresnet28',
    choices=['preactresnet18', 'preactwideresnet18', 'wideresnet28'], help='Choose architecture.')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 0)')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--train-batch-size', type=int, default=128, help='Batch size.')
parser.add_argument('--test-batch-size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-wd', type=float, default=0.0005, help='Weight decay (L2 penalty).')

# AugMix options
parser.add_argument('--augmix', type=int, default=1, metavar='S', help='aug mixup (default: 1)')
parser.add_argument('--mixture-width', default=3, type=int, help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture-depth', default=-1, type=int, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug-severity', default=3, type=int, help='Severity of base augmentation operators')
parser.add_argument('--jsd', type=int, default=1, metavar='S', help='JSD consistency loss (default: 1)')

parser.add_argument('--all-ops', '-all', action='store_true', help='Turn on all operations (+brightness,contrast,color,sharpness).')

# Noisy Feature Mixup options
parser.add_argument('--alpha', type=float, default=1.0, metavar='S', help='for mixup')
parser.add_argument('--manifold_mixup', type=int, default=1, metavar='S', help='manifold mixup (default: 0)')
parser.add_argument('--add_noise_level', type=float, default=0.5, metavar='S', help='level of additive noise')
parser.add_argument('--mult_noise_level', type=float, default=0.5, metavar='S', help='level of multiplicative noise')
parser.add_argument('--sparse_level', type=float, default=0.65, metavar='S', help='sparse noise')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(net, train_loader, optimizer, scheduler):
  """Train for one epoch."""
  net.train()
  loss_ema = 0.
  
  criterion = torch.nn.CrossEntropyLoss().to(device)
  
  for i, (images, targets) in enumerate(train_loader):
    optimizer.zero_grad()

    if args.jsd == 0:
        images = images.to(device)
        targets = targets.to(device)
      
        if args.alpha == 0.0:   
            outputs = net(images)
        else:
            outputs, targets_a, targets_b, lam = net(images, targets=targets, jsd=args.jsd,
                                                     mixup_alpha=args.alpha,
                                                      manifold_mixup=args.manifold_mixup,
                                                      add_noise_level=args.add_noise_level,
                                                      mult_noise_level=args.mult_noise_level,
                                                      sparse_level=args.sparse_level,
                                                      mode='patched_pnorm')
        
        if args.alpha>0:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)      
    
    
    elif args.jsd == 1:
      images_all = torch.cat(images, 0).to(device)
      targets = targets.to(device)     
      
      if args.alpha == 0.0:   
            logits_all = net(images_all)
      else:
            logits_all, targets_a, targets_b, lam = net(images_all, targets=targets, jsd=args.jsd, 
                                                        mixup_alpha=args.alpha,
                                                      manifold_mixup=args.manifold_mixup,
                                                      add_noise_level=args.add_noise_level,
                                                      mult_noise_level=args.mult_noise_level,
                                                      sparse_level=args.sparse_level,
                                                      mode='patched_pnorm')
        
      if args.alpha>0:
          logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
          loss = mixup_criterion(criterion, logits_clean, targets_a, targets_b, lam)
      else:
          logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
          loss = criterion(logits_clean, targets)         
      
      # JSD Loss

      p_clean, p_aug1, p_aug2 = F.softmax(
          logits_clean, dim=1), F.softmax(
              logits_aug1, dim=1), F.softmax(
                  logits_aug2, dim=1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
      p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
      loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
  return loss_ema     


def test(net, test_loader):
      """Evaluate network on given dataset."""
      net.eval()
      total_loss = 0.
      total_correct = 0
      with torch.no_grad():
        for images, targets in test_loader:
          images, targets = images.to(device), targets.to(device)
          logits = net(images)
          loss = F.cross_entropy(logits, targets)
          pred = logits.data.max(1)[1]
          total_loss += float(loss.data)
          total_correct += pred.eq(targets.data).sum().item()
    
      return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)


def main():
      torch.manual_seed(args.seed)
      np.random.seed(args.seed)
      torch.cuda.manual_seed(args.seed)
      torch.cuda.manual_seed_all(args.seed)
    
      # Load datasets
      train_transform = transforms.Compose(
          [transforms.RandomHorizontalFlip(),
           transforms.RandomCrop(32, padding=4)])
      preprocess = transforms.Compose(
          [transforms.ToTensor(),
           transforms.Normalize([0.5] * 3, [0.5] * 3)
          ])
      test_transform = preprocess
    
      if args.augmix == 0:
          train_transform = transforms.Compose(
              [transforms.RandomHorizontalFlip(),
               transforms.RandomCrop(32, padding=4),
               transforms.ToTensor(),
               transforms.Normalize([0.5] * 3, [0.5] * 3),
               ])
         

      if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(
            '../experiments/data', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(
            '../experiments/data', train=False, transform=test_transform, download=True)
        num_classes = 10
      else:
        train_data = datasets.CIFAR100(
            '../experiments/data', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(
            '../experiments/data', train=False, transform=test_transform, download=True)
        num_classes = 100
    
      if args.augmix == 1:
          train_data = AugMixDataset(train_data, preprocess, args.jsd, args)
      
      train_loader = torch.utils.data.DataLoader(
              train_data, batch_size=args.train_batch_size,
              shuffle=True, num_workers=4, pin_memory=True)          
    
      test_loader = torch.utils.data.DataLoader(
          test_data, batch_size=args.test_batch_size,
          shuffle=False, num_workers=4, pin_memory=True)
    
      # Create model
      if args.arch == 'preactresnet18':
        net = preactresnet18(num_classes=num_classes)
      elif args.arch == 'preactwideresnet18':
        net = preactwideresnet18(num_classes=num_classes)
      elif args.arch == 'wideresnet28':
          net = wideresnet28(num_classes=num_classes)
      
      # Distribute model across all visible GPUs
      net = torch.nn.DataParallel(net).to(device)
      #cudnn.benchmark = True
      optimizer = torch.optim.SGD(net.parameters(),
      args.learning_rate, momentum=args.momentum,
      weight_decay=args.decay, nesterov=True)

      start_epoch = 0
    
      scheduler = torch.optim.lr_scheduler.LambdaLR(
          optimizer,
          lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
              step, args.epochs * len(train_loader),
              1,  # lr_lambda computes multiplicative factor
              1e-6 / args.learning_rate))
      
      resume = False
      if resume:
          DESTINATION_PATH = 'experiments/trained_models/NoisyMix_' + args.dataset + '_models/'
          OUT_DIR = os.path.join(DESTINATION_PATH, f'last_arch_{args.arch}_augmix_{args.augmix}_jsd_{args.jsd}_alpha_{args.alpha}_manimixup_{args.manifold_mixup}_addn_{args.add_noise_level}_multn_{args.mult_noise_level}_seed_{args.seed}')
          checkpoint = torch.load(OUT_DIR+'.pt')
          
          net.load_state_dict(checkpoint['net_state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
          start_epoch = checkpoint['epoch'] + 1
          print(f'resuming from epoch {start_epoch}.')    
    
      best_acc = 0
      
      for epoch in range(start_epoch, args.epochs):
            
                train_loss_ema = train(net, train_loader, optimizer, scheduler)
                test_loss, test_acc = test(net, test_loader)
            
                is_best = test_acc > best_acc
                best_acc = max(test_acc, best_acc)

                DESTINATION_PATH = 'experiments/trained_models/NoisyMix_' + args.dataset + '_models/'
                OUT_DIR = os.path.join(DESTINATION_PATH, f'last_arch_{args.arch}_augmix_{args.augmix}_jsd_{args.jsd}_alpha_{args.alpha}_manimixup_{args.manifold_mixup}_addn_{args.add_noise_level}_multn_{args.mult_noise_level}_seed_{args.seed}')
                if not os.path.isdir(DESTINATION_PATH):
                            os.mkdir(DESTINATION_PATH)
                torch.save({
                        'epoch': epoch,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, OUT_DIR+'.pt')
                
                if is_best:
                  OUT_DIR = os.path.join(DESTINATION_PATH, f'best_arch_{args.arch}_augmix_{args.augmix}_jsd_{args.jsd}_alpha_{args.alpha}_manimixup_{args.manifold_mixup}_addn_{args.add_noise_level}_multn_{args.mult_noise_level}_seed_{args.seed}')
                  torch.save(net, OUT_DIR+'.pt')            
            
                print(
                    'Epoch {0:3d} | Train Loss {1:.4f} |'
                    ' Test Accuracy {2:.2f}'
                    .format((epoch + 1), train_loss_ema, 100. * test_acc))    

if __name__ == '__main__':
  main()
