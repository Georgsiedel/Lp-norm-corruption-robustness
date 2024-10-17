import re
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from skimage.util import random_noise
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import torch.distributions as dist
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_corruptions = np.array([
['uniform-linf', 0.15, False],
['uniform-l2', 5.0, False],
['uniform-l0-impulse', 0.12, True],
])
noise_patch_scale = {'lower': 1.0, 'upper': 1.0}
random_noise_dist = 'uniform'

def get_image_mask(batch, noise_patch_scale=1.0, ratio=[0.3, 3.3]):
    """Get image mask for Patched Noise (see Random Erasing and Patch Gaussian papers).
    Args:
        batch (Tensor): batch of images to be masked.
        noise_patch_lower_scale (sequence): Lower bound for range of proportion of masked area against input image. Upper bound is 1.0
        ratio (sequence): range of aspect ratio of masked area.

        This, with the lines below commented out and ratio [1.0,1.0], is the original patch gaussian implementation
        with square patches the center of which can be anywhere on the image, so the patch may be outside. Consequently,
        with the lines commented out, you may pass upper scale values higher then 1.0, as 1.0 may not always cover the full image.
    """
    img_c, img_h, img_w = batch.shape[-3], batch.shape[-2], batch.shape[-1]
    area = img_h * img_w

    if noise_patch_scale == 1.0 or noise_patch_scale == [1.0, 1.0]:
        return torch.ones(batch.size(), dtype=torch.bool, device=device)
    else:
        patched_area = area * torch.empty(1).uniform_(noise_patch_scale[0], noise_patch_scale[1]).item()

    log_ratio = torch.log(torch.tensor(ratio))
    aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

    h = int(round(math.sqrt(patched_area * aspect_ratio)))
    w = int(round(math.sqrt(patched_area / aspect_ratio)))
    # if h > img_h:
    #    h = img_h
    #    w = int(round(img_w * patched_area / area)) #reset patched area ratio when patch needs to be cropped due to aspect ratio
    # if w > img_w:
    #    w = img_w
    #    h = int(round(img_h * patched_area / area)) #reset patched area ratio when patch needs to be cropped due to aspect ratio
    i = torch.randint(0, img_h + 1, size=(1,)).item()
    j = torch.randint(0, img_w + 1, size=(1,)).item()
    lower_y = int(round(i - h / 2 if i - h / 2 >= 0 else 0))
    higher_y = int(round(i + h / 2 if i + h / 2 <= img_h else img_h))
    lower_x = int(round(j - w / 2 if j - w / 2 >= 0 else 0))
    higher_x = int(round(j + w / 2 if j + w / 2 <= img_w else img_w))
    mask = torch.zeros(batch.size(), dtype=torch.bool, device=device)
    mask[:, :, lower_y: higher_y, lower_x: higher_x] = True

    return mask

def apply_lp_corruption(batch, minibatchsize, combine_train_corruptions, corruptions, concurrent_combinations,
                        noise_patch_scale=1.0, random_noise_dist=None, factor = 1):
    minibatches = batch.view(-1, minibatchsize, batch.size()[1], batch.size()[2], batch.size()[3])

    for id, minibatch in enumerate(minibatches):
        if combine_train_corruptions == True:
            corruptions_list = random.sample(list(corruptions), k=concurrent_combinations)
            for x, (noise_type, train_epsilon, max) in enumerate(corruptions_list):
                bool_max = max == True
                noisy_minibatch = sample_lp_corr_batch(noise_type, train_epsilon, minibatch, bool_max,
                                                       random_noise_dist, factor)
        else:
            bool_max = corruptions[2] == True
            noisy_minibatch = sample_lp_corr_batch(corruptions[0], corruptions[1], minibatch, bool_max,
                                                   random_noise_dist, factor)
        patch_mask = get_image_mask(minibatch, noise_patch_scale=noise_patch_scale, ratio=[1.0, 1.0])
        final_minibatch = torch.where(patch_mask, noisy_minibatch, minibatch)
        minibatches[id] = final_minibatch
    batch = minibatches.view(-1, batch.size()[1], batch.size()[2], batch.size()[3])

    return batch

def sample_lp_corr_batch(noise_type, epsilon, batch, density_distribution_max, random_noise_dist = None, factor = 1):
    with torch.cuda.device(0):
        corruption = torch.zeros(batch.size(), dtype=torch.float16)
        if random_noise_dist == 'uniform':
            random_factor = torch.rand(1).item()
        elif random_noise_dist == 'beta':
            random_factor = np.random.beta(2, 5)
        else:
            random_factor = 1

        random_factor = random_factor * factor

        if noise_type == 'uniform-linf':
            if density_distribution_max == True:  # sample on the hull of the norm ball
                rand = np.random.random(batch.shape)
                sign = np.where(rand < 0.5, -1, 1)
                corruption = torch.from_numpy(sign * float(epsilon) * random_factor)
            else: #sample uniformly inside the norm ball
                #corruption = torch.cuda.FloatTensor(batch.shape).uniform_(-epsilon, epsilon)
                corruption = torch.rand(batch.shape, device=device, dtype=torch.float16) * float(epsilon) * random_factor
        elif noise_type == 'gaussian': #note that the option density_distribution_max = False here does not do anything
            #corruption = torch.cuda.FloatTensor(batch.shape).normal_(0, epsilon)
            corruption = torch.randn(batch.shape, device=device, dtype=torch.float16) * float(epsilon) * random_factor
        elif noise_type == 'uniform-l0-impulse':
            num_dimensions = torch.numel(batch[0])
            num_pixels = int(num_dimensions * float(epsilon) * random_factor)
            lower_bounds = torch.arange(0, batch.size(0) * num_dimensions, num_dimensions, device=device)
            upper_bounds = torch.arange(num_dimensions, (batch.size(0) + 1) * num_dimensions, num_dimensions, device=device)
            indices = torch.cat([torch.randint(l, u, (num_pixels,), device=device) for l, u in zip(lower_bounds, upper_bounds)])
            mask = torch.full(batch.size(), False, dtype=torch.bool, device=device)
            mask.view(-1)[indices] = True

            if density_distribution_max == True:
                random_numbers = torch.randint(2, size=batch.size(), dtype=torch.float16, device=device) #* 2 - 1
            else:
                #random_numbers = torch.cuda.FloatTensor(batch.shape).uniform_(0, 1)
                random_numbers = torch.rand(batch.shape, device=device, dtype=torch.float16) #* 2 - 1
            batch_corr = torch.where(mask, random_numbers, batch)

            return batch_corr

        elif 'uniform-l' in noise_type:  #Calafiore1998: Uniform Sample Generation in lp Balls for Probabilistic Robustness Analysis
            img_corr = torch.zeros(batch[0].size(), dtype=torch.float16, device=device)
            #number of dimensions
            d = img_corr.numel()
            # extract Lp-number from args.noise variable
            lp = [float(x) for x in re.findall(r'-?\d+\.?\d*', noise_type)][0]
            u = dist.Gamma(1 / lp, 1).sample(img_corr.shape).to(device)
            u = u ** (1 / lp)
            #sign = torch.where(torch.cuda.FloatTensor(img_corr.shape).uniform_(0, 1) < 0.5, -1, 1)
            sign = torch.where(torch.rand(batch.shape, device=device, dtype=torch.float16) < 0.5, -1, 1)
            norm = torch.sum(abs(u) ** lp) ** (1 / lp)  # scalar, norm samples to lp-norm-sphere
            if density_distribution_max == True:
                r = 1
            else:  # uniform density distribution
                r = dist.Uniform(0, 1).sample() ** (1.0 / d)
            img_corr = float(epsilon) * random_factor * r * u * sign / norm #image-sized corruption, epsilon * random radius * random array / normed
            corruption = img_corr.expand(batch.size()).to(device)

        elif noise_type == 'standard':
            return batch
        else:
            print('Unknown type of noise')

    corruption = corruption.to(device)
    corrupted_batch = torch.clamp(batch + corruption, 0, 1)
    return corrupted_batch
