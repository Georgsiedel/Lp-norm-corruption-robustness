import re
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from skimage.util import random_noise
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.distributions as dist

def sample_lp_corr_batch(noise_type, epsilon, batch, density_distribution_max):
    img_corr = torch.zeros(batch[0].size(), dtype=torch.float32)

    if noise_type == 'uniform-linf':
        if density_distribution_max == True:  # sample on the hull of the norm ball
            rand = np.random.random(img_corr.shape)
            sign = np.where(rand < 0.5, -1, 1)
            img_corr = sign * epsilon
            img_corr = torch.from_numpy(img_corr)
        else: #sample uniformly inside the norm ball
            img_corr = dist.Uniform(-epsilon, epsilon).sample()
    elif noise_type == 'uniform-linf-brightness': #only max-distribution, every pixel gets same manipulation
        img_corr = random.choice([-epsilon, epsilon])
    elif noise_type == 'gaussian': #note that this has no option for density_distribution=max
        var = epsilon * epsilon
        img_corr = torch.tensor(random_noise(img_corr, mode='gaussian', mean=0, var=var, clip=True))
    elif noise_type == 'uniform-l0-salt-pepper': #note that this has no option for density_distribution=max
        num_pixels = round(epsilon * torch.numel(img_corr[0]))
        pixels = random.sample(range(torch.numel(img_corr[0])), num_pixels)
        for pixel in pixels:
            max_pixel = random.choice([0, 1])
            img_corr[0].view(-1)[pixel] = max_pixel
            img_corr[1].view(-1)[pixel] = max_pixel
            img_corr[2].view(-1)[pixel] = max_pixel
    elif noise_type == 'uniform-l0-impulse':
        num_pixels = round(epsilon * torch.numel(img_corr))
        if density_distribution_max == True:
            pixels = random.sample(range(torch.numel(img_corr)), num_pixels)
            for pixel in pixels:
                img_corr.view(-1)[pixel] = random.choice([-1, 1])
        else:
            for id, img in enumerate(batch):
                pixels = []
                x = torch.numel(img_corr)
                pixels.append(random.sample(range(x*id, x*(id+1)), num_pixels))
            for pixel in pixels:
                batch.view(-1)[pixel] = random.randint(0, 255) / 255
    elif 'uniform-l' in noise_type:  #Calafiore1998: Uniform Sample Generation in lp Balls for Probabilistic Robustness Analysis
        d = len(img_corr.ravel())
        lp = [float(x) for x in re.findall(r'-?\d+\.?\d*', noise_type)]  # extract Lp-number from args.noise variable
        lp = lp[0]
        u = np.random.gamma(1 / lp, 1, size=(np.array(img_corr).shape))  # image-sized array of Laplace-distributed random variables (distribution beta factor equalling Lp-norm)
        u = u ** (1 / lp)
        rand = np.random.random(np.array(img_corr).shape)
        sign = np.where(rand < 0.5, -1, 1)
        norm = np.sum(abs(u) ** lp) ** (1 / lp) # scalar, norm samples to lp-norm-sphere
        if density_distribution_max == True:
            r = 1 # 1 to leave the sampled points on the hull of the norm ball, to sample uniformly within use this: np.random.random() ** (1.0 / d)
        else: #uniform density distribution
            r = np.random.random() ** (1.0 / d)
        img_corr = epsilon * r * u * sign / norm  #image-sized corruption, epsilon * random radius * random array / normed
        img_corr = torch.from_numpy(img_corr)
    elif noise_type == 'standard':
        pass
    else:
        print('Unknown type of noise')

    batch_corr = batch + img_corr.unsqueeze(0)
    batch_corr = np.clip(batch_corr, 0, 1)  # clip values below 0 and over 1
    return batch_corr

def sample_lp_corr_img(noise_type, epsilon, img, density_distribution_max):
    d = len(img.ravel())
    if epsilon == 0:
        img_corr = img
    else:
        if noise_type == 'uniform-linf':
            if density_distribution_max == True:  # sample on the hull of the norm ball
                rand = np.random.random(np.array(img).shape)
                sign = np.where(rand < 0.5, -1, 1)
                img_corr = img + (sign * epsilon)
            else: #sample uniformly inside the norm ball
                img_corr = dist.Uniform(img - epsilon, img + epsilon).sample()
            img_corr = np.clip(img_corr, 0, 1) # clip values below 0 and over 1
        elif noise_type == 'uniform-linf-brightness': #only max-distribution, every pixel gets same manipulation
            img_corr = img
            img_corr = random.choice([img_corr - epsilon, img_corr + epsilon])
            img_corr = np.clip(img_corr, 0, 1) # clip values below 0 and over 1
        elif noise_type == 'gaussian': #note that this has no option for density_distribution=max
            var = epsilon * epsilon
            img_corr = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=var, clip=True))
        elif noise_type == 'uniform-l0-salt-pepper': #note that this has no option for density_distribution=max
            img_corr = img
            num_pixels = round(epsilon * torch.numel(img_corr[0]))
            pixels = random.sample(range(torch.numel(img_corr[0])), num_pixels)
            for pixel in pixels:
                max_pixel = random.choice([0, 1])
                img_corr[0].view(-1)[pixel] = max_pixel
                img_corr[1].view(-1)[pixel] = max_pixel
                img_corr[2].view(-1)[pixel] = max_pixel
        elif noise_type == 'uniform-l0-impulse':
            img_corr = img
            num_pixels = round(epsilon * torch.numel(img_corr))
            pixels = random.sample(range(torch.numel(img_corr)), num_pixels)
            if density_distribution_max == True:
                for pixel in pixels:
                    img_corr.view(-1)[pixel] = random.choice([0, 1])
            else:
                for pixel in pixels:
                    img_corr.view(-1)[pixel] = random.randint(0, 255) / 255
        elif 'uniform-l' in noise_type:  #Calafiore1998: Uniform Sample Generation in lp Balls for Probabilistic Robustness Analysis
            lp = [float(x) for x in re.findall(r'-?\d+\.?\d*', noise_type)]  # extract Lp-number from args.noise variable
            lp = lp[0]
            u = np.random.gamma(1/lp, 1, size=(np.array(img).shape))  # image-sized array of Laplace-distributed random variables (distribution beta factor equalling Lp-norm)
            u = u ** (1/lp)
            rand = np.random.random(np.array(img).shape)
            sign = np.where(rand < 0.5, -1, 1)
            norm = np.sum(abs(u) ** lp) ** (1 / lp)  # scalar, norm samples to lp-norm-sphere
            if density_distribution_max == True:
                r = 1 # 1 to leave the sampled points on the hull of the norm ball, to sample uniformly within use this: np.random.random() ** (1.0 / d)
            else: #uniform density distribution
                r = np.random.random() ** (1.0 / d)
            corr = epsilon * r * u * sign / norm  #image-sized corruption, epsilon * random radius * random array / normed
            img_corr = img + corr  # construct corrupted image by adding sampled noise
            img_corr = np.clip(img_corr, 0, 1) #clip values below 0 and over 1
        else:
            img_corr = img
            print('Unknown type of noise')
    return img_corr

#Sample 3 images in original form and with a chosen maximum corruption of a chose Lp norm.
#Use this e.g. to estimate maximum Lp-corruptions which should not change the class, or which are quasi-imperceptible.
def sample_lp_corr_visualization(n_images = 3, seed = -1, noise_type = 'uniform-linf', epsilon = 8/255, density_distribution = "max", dataset = 'CIFAR10'):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == 'ImageNet' or dataset == 'TinyImageNet':
        trainset = torchvision.datasets.ImageFolder(root=f'./data/{dataset}/train', transform=transform)
        truncated_set = torch.utils.data.Subset(trainset, range(0, 100000, 5000))
        loader = torch.utils.data.DataLoader(truncated_set, batch_size=1, shuffle=False)
    else:
        load_helper = getattr(datasets, dataset)
        trainset = load_helper(root='./data', train=True, download=True, transform=transform)
        truncated_set = torch.utils.data.Subset(trainset, range(0, 100000, 5000))
        loader = torch.utils.data.DataLoader(truncated_set, batch_size=1, shuffle=False)

    fig, axs = plt.subplots(n_images, 2)
    j = seed
    for i in range(n_images):
        if seed == -1:
            j = random.randint(0, len(loader)) # selecting random images from the train dataset
        for id, (input, target) in enumerate(loader):
            if j == id:
                image = input
                corrupted_image = input
                break
        image = torch.squeeze(image)
        corrupted_image = torch.squeeze(corrupted_image)
        img_corr = sample_lp_corr_img(noise_type, epsilon, corrupted_image, density_distribution)
        corrupted_image = corrupted_image + img_corr  # construct corrupted image by adding sampled noise
        corrupted_image = np.clip(corrupted_image, 0, 1)  # clip values below 0 and over 1
        image = image.permute(1, 2, 0)
        corrupted_image = corrupted_image.permute(1, 2, 0)
        axs[i, 0].imshow(image)
        axs[i, 1].imshow(corrupted_image)
        j = j+1
    return fig

if __name__ == '__main__':
    fig = sample_lp_corr_img(3, -1, 'uniform-linf', 0.01, "max", "TinyImageNet")
    plt.show()