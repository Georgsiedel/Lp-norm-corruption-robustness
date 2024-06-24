from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import torch
import torch.cuda.amp
import torchvision
import torchvision.transforms as transforms
import math
from typing import Tuple
from torch import Tensor
from torchvision.transforms import functional as F

from experiments.sample_lp_corruption import sample_lp_corr_batch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("Please provide a valid positive value for the num_classes.")
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = F.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

def apply_mixing_functions(inputs, targets, mixup_alpha, cutmix_alpha, num_classes):
    mixes = []
    if mixup_alpha > 0.0:
        mixes.append(RandomMixup(num_classes, p=1.0, alpha=mixup_alpha))
    if cutmix_alpha > 0.0:
        mixes.append(RandomCutmix(num_classes, p=1.0, alpha=cutmix_alpha))
    if mixes:
        mixupcutmix = torchvision.transforms.RandomChoice(mixes)
        inputs, targets = mixupcutmix(inputs, targets)
    return inputs, targets

def normalize(inputs, dataset):
    if dataset == 'CIFAR10':
        inputs = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))(inputs)
    elif dataset == 'CIFAR100':
        inputs = transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))(
            inputs)
    elif (dataset == 'ImageNet' or dataset == 'TinyImageNet'):
        inputs = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(inputs)
    else:
        print('no normalization values set for this dataset')
    return inputs

class AugmentedDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform augmentations and allow robust loss functions."""

  def __init__(self, dataset, transform_train, transform_valid, robust_samples=0):
    self.dataset = dataset
    self.augment = transform_train
    self.robust_samples = robust_samples
    self.preprocess =transform_valid

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.robust_samples == 0:
      return self.augment(x), y
    elif self.robust_samples == 1:
      im_tuple = (self.preprocess(x), self.augment(x))
      return im_tuple, y
    elif self.robust_samples == 2:
      im_tuple = (self.preprocess(x), self.augment(x), self.augment(x))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)

def apply_augstrat(batch, train_aug_strat):
    for id, img in enumerate(batch):
        img = img * 255.0
        img = img.type(torch.uint8)
        tf = getattr(transforms, train_aug_strat)
        img = tf()(img)
        img = img.type(torch.float32) / 255.0
        batch[id] = img

    return batch

def get_image_mask(batch, noise_patch_lower_scale=1.0, ratio=[0.3, 3.3]):
        """Get image mask for Patched Noise (see e.g. Patch Gaussian paper).
        Args:
            batch (Tensor): batch of images to be masked.
            noise_patch_lower_scale (sequence): Lower bound for range of proportion of masked area against input image. Upper bound is 1.0
            ratio (sequence): range of aspect ratio of masked area.
        """
        if noise_patch_lower_scale == 1.0:
            return torch.ones(batch.size(), dtype=torch.bool, device=device)

        img_c, img_h, img_w = batch.shape[-3], batch.shape[-2], batch.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))

        patched_area = area * torch.empty(1).uniform_(noise_patch_lower_scale, 1.0).item()
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        h = int(round(math.sqrt(patched_area * aspect_ratio)))
        w = int(round(math.sqrt(patched_area / aspect_ratio)))
        if h > img_h:
            h = img_h
            w = int(round(img_w * patched_area / area)) #reset patched area ratio when patch needs to be cropped due to aspect ratio
        if w > img_w:
            w = img_w
            h = int(round(img_h * patched_area / area)) #reset patched area ratio when patch needs to be cropped due to aspect ratio
        i = torch.randint(0, img_h + 1, size=(1,)).item()
        j = torch.randint(0, img_w + 1, size=(1,)).item()
        lower_y = int(round(i-h/2 if i-h/2 >= 0 else 0))
        higher_y = int(round(i+h/2 if i+h/2 <= img_h else img_h))
        lower_x = int(round(j - w / 2 if j - w / 2 >= 0 else 0))
        higher_x = int(round(j+w/2 if j+w/2 <= img_w else img_w))
        mask = torch.zeros(batch.size(), dtype=torch.bool, device=device)
        mask[:,:,lower_y : higher_y, lower_x : higher_x] = True

        return mask

def apply_lp_corruption(batch, minibatchsize, combine_train_corruptions, train_corruptions, concurrent_combinations, noise_patch_lower_scale=1.0):

    minibatches = batch.view(-1, minibatchsize, batch.size()[1], batch.size()[2], batch.size()[3])

    for id, minibatch in enumerate(minibatches):
        if combine_train_corruptions == True:
            corruptions_list = random.sample(list(train_corruptions), k=concurrent_combinations)
            for x, (noise_type, train_epsilon, max) in enumerate(corruptions_list):
                train_epsilon = float(train_epsilon)
                noisy_minibatch = sample_lp_corr_batch(noise_type, train_epsilon, minibatch, max)
        else:
            noisy_minibatch = sample_lp_corr_batch(train_corruptions[0], train_corruptions[1], minibatch, train_corruptions[2])
        patch_mask = get_image_mask(minibatch, noise_patch_lower_scale=noise_patch_lower_scale, ratio=[1.0, 1.0])
        final_minibatch = torch.where(patch_mask, noisy_minibatch, minibatch)
        minibatches[id] = final_minibatch
    batch = minibatches.view(-1, batch.size()[1], batch.size()[2], batch.size()[3])

    return batch

def create_transforms(dataset, aug_strat_check, train_aug_strat, RandomEraseProbability):
    # list of all data transformations used
    t = transforms.ToTensor()
    c32 = transforms.RandomCrop(32, padding=4)
    c64 = transforms.RandomCrop(64, padding=8)
    flip = transforms.RandomHorizontalFlip()
    r256 = transforms.Resize(256, antialias=True)
    c224 = transforms.CenterCrop(224)
    rrc224 = transforms.RandomResizedCrop(224, antialias=True)
    re = transforms.RandomErasing(p=RandomEraseProbability)
    tf = getattr(transforms, train_aug_strat)

    # transformations of validation set
    transforms_valid = transforms.Compose([t])
    if dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'TinyImageNet':
        transforms_valid = transforms.Compose([transforms_valid])
    elif dataset == 'ImageNet':
        transforms_valid = transforms.Compose([transforms_valid, r256, c224])

    # transformations of training set
    transforms_train = transforms.Compose([flip])
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        transforms_train = transforms.Compose([transforms_train, c32])
    elif dataset == 'TinyImageNet':
        transforms_train = transforms.Compose([transforms_train, c64])
    elif dataset == 'ImageNet':
        transforms_train = transforms.Compose([transforms_train, rrc224])
    if aug_strat_check == True:
        transforms_train = transforms.Compose([transforms_train, tf()])

    transforms_train = transforms.Compose([transforms_train, t, re])


    return transforms_train, transforms_valid
