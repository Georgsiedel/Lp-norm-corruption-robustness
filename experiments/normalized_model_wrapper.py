from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import experiments.models as low_dim_models

def create_normalized_model_wrapper(dataset, modeltype):
    if dataset == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1)
    elif dataset == 'CIFAR100':
        mean = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(1, 3, 1, 1)
        std = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    model = getattr(low_dim_models, modeltype)

    class Normalized_Model_Wrapper(model):
        def __init__(self, **kwargs):
            super(Normalized_Model_Wrapper, self).__init__(**kwargs)
            self.register_buffer('mu', mean)
            self.register_buffer('sigma', std)

        def forward(self, x):
            x = (x - self.mu) / self.sigma
            return super(Normalized_Model_Wrapper, self).forward(x)

    return Normalized_Model_Wrapper