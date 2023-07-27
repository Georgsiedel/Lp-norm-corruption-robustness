from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import time
import torch
import torchvision
import torchvision.transforms as transforms
from skimage.util import random_noise
import numpy as np
import random
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torch.distributions as dist
x_min = torch.tensor([0, 0, 0])
x_max = torch.tensor([1, 1, 1])

def sample_lp_corr(noise_type, epsilon, img, density_distribution):
    d = len(img.ravel())
    if epsilon == 0:
        img_corr = img
    else:
        if noise_type == 'uniform-linf':
            if density_distribution == 'max':  # sample on the hull of the norm ball
                rand = np.random.random(img.shape)
                sign = np.where(rand < 0.5, -1, 1)
                img_corr = img + (sign * epsilon)
            else: #sample uniformly inside the norm ball
                img_corr = dist.Uniform(img - epsilon, img + epsilon).sample()
            img_corr = np.ma.clip(img_corr, 0, 1) # clip values below 0 and over 1
        elif noise_type == 'uniform-linf-brightness': #only max-distribution, every pixel gets same manipulation
            img_corr = img
            img_corr = random.choice([img_corr - epsilon, img_corr + epsilon])
            img_corr = np.ma.clip(img_corr, 0, 1) # clip values below 0 and over 1
        elif noise_type == 'gaussian': #note that this has no option for density_distribution=max
            var = epsilon * epsilon
            img_corr = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=var, clip=True))
        elif noise_type == 'uniform-l0-salt-pepper': #note that this has no option for density_distribution=max
            img_corr = img
            pixels = []
            for j in range(round(epsilon * torch.numel(img_corr[0]))):
                pixels.append(random.randint(0, torch.numel(img_corr[0])))
            for pixel in pixels:
                max_pixel = random.choice([0, 1])
                img_corr[0].view(-1)[pixel-1] = max_pixel
                img_corr[1].view(-1)[pixel-1] = max_pixel
                img_corr[2].view(-1)[pixel-1] = max_pixel
        elif noise_type == 'uniform-l0-impulse':
            img_corr = img
            pixels = []
            for j in range(round(epsilon * torch.numel(img_corr))):
                pixels.append(random.randint(0, torch.numel(img_corr)))
            if density_distribution == 'max':
                for pixel in pixels:
                    img_corr.view(-1)[pixel-1] = random.choice([0, 1])
            else:
                for pixel in pixels:
                    img_corr.view(-1)[pixel - 1] = random.randint(0, 255) / 255
        elif 'uniform-l' in noise_type:  #Calafiore1998: Uniform Sample Generation in lp Balls for Probabilistic Robustness Analysis
            lp = [float(x) for x in re.findall(r'-?\d+\.?\d*', noise_type)]  # extract Lp-number from args.noise variable
            lp = lp[0]
            u = np.random.laplace(0, 1 / lp, size=(np.array(img).shape))  # image-sized array of Laplace-distributed random variables (distribution beta factor equalling Lp-norm)
            norm = np.sum(abs(u) ** lp) ** (1 / lp)  # scalar, norm samples to lp-norm-sphere
            if density_distribution == 'max':
                r = 1 # 1 to leave the sampled points on the hull of the norm ball, to sample uniformly within use this: np.random.random() ** (1.0 / d)
            else: #uniform density distribution
                r = np.random.random() ** (1.0 / d)
            corr = epsilon * r * u / norm  #image-sized corruption, epsilon * random radius * random array / normed
            img_corr = img + corr  # construct corrupted image by adding sampled noise
            img_corr = np.ma.clip(img_corr, 0, 1) #clip values below 0 and over 1
        else:
            img_corr = img
            print('Unknown type of noise')
    return img_corr


#Sample 3 images in original form and with a chosen maximum corruption of a chose Lp norm.
#Use this e.g. to estimate maximum Lp-corruptions which should not change the class, or which are quasi-imperceptible.
def sample_corr_img(n_images = 3, random = False, noise_type = 'uniform-linf', epsilon = 8/255, density_distribution = "max"):
    transform_train = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
    j = 1
    fig, axs = plt.subplots(n_images, 2)
    for i in range(n_images):
        if random == True:
            j = random.randint(0, len(trainloader)) # selecting random images from the train dataset
        for id, (input, target) in enumerate(trainloader):
            if j == id:
                image = input
                corrupted_image = input
                break
        image = torch.squeeze(image)
        corrupted_image = torch.squeeze(corrupted_image)
        corrupted_image = sample_lp_corr(noise_type, epsilon, corrupted_image, density_distribution)
        image = image.permute(1, 2, 0)
        corrupted_image = corrupted_image.permute(1, 2, 0)
        axs[i, 0].imshow(image)
        axs[i, 1].imshow(corrupted_image)
        j = j+1
    return fig

if __name__ == '__main__':
    fig = sample_corr_img(3, False, 'uniform-l2', 1, "max")
    plt.show()