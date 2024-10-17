import numpy as np
import torch
import src.p_corruption as p_corruption
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _noise(x, add_noise_level=0.0, mult_noise_level=0.0, sparse_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    with torch.cuda.device(0):
        if add_noise_level > 0.0:
            var = torch.var(x)**0.5
            add_noise = add_noise_level * np.random.beta(2, 5) * torch.empty(x.shape, dtype=torch.float16, device=device).normal_()
            #torch.clamp(add_noise, min=-(2*var), max=(2*var), out=add_noise) # clamp
            sparse = torch.empty(x.shape, dtype=torch.float16, device=device).uniform_()
            add_noise[sparse<sparse_level] = 0
        if mult_noise_level > 0.0:
            mult_noise = mult_noise_level * np.random.beta(2, 5) * (2*torch.empty(x.shape, dtype=torch.float16, device=device).uniform_()-1) + 1 
            sparse = torch.empty(x.shape, dtype=torch.float16, device=device).uniform_()
            mult_noise[sparse<sparse_level] = 1.0

            
    return mult_noise * x + add_noise      

def normalize_batch(tensor, mean, std):
    """
    Normalize a batch of images channel-wise.
    
    Args:
        tensor (torch.Tensor): Batch of images (shape: (batch_size, num_channels, height, width))
        mean (list or torch.Tensor): Mean for each channel (e.g., [mean_R, mean_G, mean_B])
        std (list or torch.Tensor): Standard deviation for each channel (e.g., [std_R, std_G, std_B])
        
    Returns:
        torch.Tensor: Normalized batch of images
    """
    # Convert mean and std to tensors if they are not already
    mean = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)  # Shape: (1, num_channels, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)    # Shape: (1, num_channels, 1, 1)

    # Normalize the tensor
    return (tensor - mean) / std

def do_noisy_mixup(x, y, jsd=0, alpha=0.0, add_noise_level=0.0, mult_noise_level=0.0, sparse_level=0.0, mode='standard'):
    lam = np.random.beta(alpha, alpha) if alpha > 0.0 else 1.0
    
    if jsd==0:
        index = torch.randperm(x.size()[0]).to(device)
        x = lam * x + (1 - lam) * x[index]
        if mode == 'patched_standard':
            mask = p_corruption.get_image_mask(x, noise_patch_scale=[list(p_corruption.noise_patch_scale.values())[0], list(p_corruption.noise_patch_scale.values())[1]], ratio=[1.0, 1.0])
            x_noisy = _noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level, sparse_level=sparse_level)
            x = torch.where(mask, x_noisy, x)
        elif mode == 'patched_pnorm':

            #mean = [-1.0] * 3
            #std = [2.0] * 3
            #x = normalize_batch(x, mean, std)

            x = p_corruption.apply_lp_corruption(x, 
                        minibatchsize=8, 
                        combine_train_corruptions=True, 
                        corruptions=p_corruption.train_corruptions, 
                        concurrent_combinations=1, 
                        noise_patch_scale=[list(p_corruption.noise_patch_scale.values())[0], list(p_corruption.noise_patch_scale.values())[1]],
                        random_noise_dist=p_corruption.random_noise_dist,
                        factor=1)
            
            #mean = [0.5] * 3
            #std = [0.5] * 3
            #x = normalize_batch(x, mean, std)

        else:
            x = _noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level, sparse_level=sparse_level)
    else:
        kk = 0
        q = int(x.shape[0]/3)
        index = torch.randperm(q).to(device)
    
        for i in range(1,4):
            x[kk:kk+q] = lam * x[kk:kk+q] + (1 - lam) * x[kk:kk+q][index]
            if mode == 'patched_standard':
                mask = p_corruption.get_image_mask(x[kk:kk+q], noise_patch_scale=[list(p_corruption.noise_patch_scale.values())[0], list(p_corruption.noise_patch_scale.values())[1]], ratio=[1.0, 1.0])
                x_noisy = _noise(x[kk:kk+q], add_noise_level=add_noise_level, mult_noise_level=mult_noise_level, sparse_level=sparse_level)
                x[kk:kk+q] = torch.where(mask, x_noisy, x[kk:kk+q])
            elif mode == 'patched_pnorm':
                
                #mean = [-1.0] * 3
                #std = [2.0] * 3
                #x[kk:kk+q] = normalize_batch(x[kk:kk+q], mean, std)

                x[kk:kk+q] = p_corruption.apply_lp_corruption(x[kk:kk+q], 
                        minibatchsize=8, 
                        combine_train_corruptions=True, 
                        corruptions=p_corruption.train_corruptions, 
                        concurrent_combinations=1, 
                        noise_patch_scale=[list(p_corruption.noise_patch_scale.values())[0], list(p_corruption.noise_patch_scale.values())[1]],
                        random_noise_dist=p_corruption.random_noise_dist,
                        factor = i)
                
                #mean = [0.5] * 3
                #std = [0.5] * 3
                #x[kk:kk+q] = normalize_batch(x[kk:kk+q], mean, std)
                
            else:
                x[kk:kk+q] = _noise(x[kk:kk+q], add_noise_level=add_noise_level*i, mult_noise_level=mult_noise_level, sparse_level=sparse_level)
            kk += q
     
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
