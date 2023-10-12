import numpy as np
import torchvision.models.mobilenet

train_corruptions = np.array([
['uniform-linf', 0.005, False],
['uniform-linf', 0.01, False],
['uniform-linf', 0.02, False],
['uniform-linf', 0.03, False],
['uniform-linf', 0.04, False],
['uniform-l0.5', 25000.0, False],
['uniform-l0.5', 50000.0, False],
['uniform-l0.5', 75000.0, False],
['uniform-l0.5', 100000.0, False],
['uniform-l0.5', 150000.0, False],
['uniform-l1', 12.5, False],
['uniform-l1', 25.0, False],
['uniform-l1', 37.5, False],
['uniform-l1', 50.0, False],
['uniform-l1', 75.0, False],
['uniform-l2', 0.25, False],
['uniform-l2', 0.5, False],
['uniform-l2', 0.75, False],
['uniform-l2', 1.0, False],
['uniform-l2', 1.5, False],
['uniform-l5', 0.03, False],
['uniform-l5', 0.06, False],
['uniform-l5', 0.1, False],
['uniform-l5', 0.15, False],
['uniform-l5', 0.2, False],
['uniform-l10', 0.02, False],
['uniform-l10', 0.03, False],
['uniform-l10', 0.05, False],
['uniform-l10', 0.07, False],
['uniform-l10', 0.1, False],
['uniform-l50', 0.01, False],
['uniform-l50', 0.02, False],
['uniform-l50', 0.03, False],
['uniform-l50', 0.04, False],
['uniform-l50', 0.06, False],
['uniform-l200', 0.01, False],
['uniform-l200', 0.02, False],
['uniform-l200', 0.03, False],
['uniform-l200', 0.04, False],
['uniform-l200', 0.05, False],
['uniform-l0-impulse', 0.005, True],
['uniform-l0-impulse', 0.01, True],
['uniform-l0-impulse', 0.015, True],
['uniform-l0-impulse', 0.02, True],
['uniform-l0-impulse', 0.03, True],
])

batchsize = 384
dataset = 'CIFAR100' #ImageNet #CIFAR100 #CIFAR10 #TinyImageNet
if dataset == 'CIFAR10':
    num_classes = 10
    pixel_factor = 1
elif dataset == 'CIFAR100':
    num_classes = 100
    pixel_factor = 1
elif dataset == 'ImageNet':
    num_classes = 1000
elif dataset == 'TinyImageNet':
    num_classes = 200
    pixel_factor = 2
normalize = False
validontest = True
lrschedule = 'CosineAnnealingWarmRestarts'
learningrate = 0.1
epochs = 150
lrparams = {'T_0': 10, 'T_mult': 2}
warmupepochs = 0
earlystop = False
earlystopPatience = 15
optimizer = 'SGD'
optimizerparams = {'momentum': 0.9, 'weight_decay': 5e-4}
number_workers = 1
modeltype = 'DenseNet201_12'
modelparams = {}
resize = False
aug_strat_check = False
train_aug_strat = 'TrivialAugmentWide' #TrivialAugmentWide, RandAugment, AutoAugment, AugMix
jsd_loss = False
lossparams = {'num_splits': 3, 'alpha': 12, 'smoothing': 0.0}
mixup_alpha = 0.2 #default 0.2 #If both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance
cutmix_alpha = 0.0 # default 1.0 #If both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance
RandomEraseProbability = 0.0

combine_train_corruptions = True #augment the train dataset with all corruptions
concurrent_combinations = 1 #only has an effect if combine_train_corruption is True

if combine_train_corruptions:
    model_count = 1
else:
    model_count = train_corruptions.shape[0]

#define train and test corruptions:
#define noise type (first column): 'gaussian', 'uniform-l0-impulse', 'uniform-l0-salt-pepper', 'uniform-linf'. also: all positive numbers p>0 for uniform Lp possible: 'uniform-l1', 'uniform-l2', ...
#define intensity (second column): max.-distance of random perturbations for model training and evaluation (gaussian: std-dev; l0: proportion of pixels corrupted; lp: epsilon)
#define whether density_distribution=max (third column) is True (sample only maximum intensity values) or False (uniformly distributed up to maximum intensity)
test_corruptions = np.array([
['standard', 0.0, False],
['uniform-linf', 0.005, False],
['uniform-linf', 0.01, False],
['uniform-linf', 0.02, False],
['uniform-linf', 0.03, False],
['uniform-linf', 0.04, False],
['uniform-linf', 0.06, False],
['uniform-linf', 0.08, False],
['uniform-linf', 0.1, False],
['uniform-linf', 0.12, False],
['uniform-linf', 0.15, False],
['uniform-l0.5', 25000.0, False],
['uniform-l0.5', 50000.0, False],
['uniform-l0.5', 75000.0, False],
['uniform-l0.5', 100000.0, False],
['uniform-l0.5', 150000.0, False],
['uniform-l0.5', 200000.0, False],
['uniform-l0.5', 250000.0, False],
['uniform-l0.5', 300000.0, False],
['uniform-l0.5', 350000.0, False],
['uniform-l0.5', 400000.0, False],
['uniform-l1', 12.5, False],
['uniform-l1', 25.0, False],
['uniform-l1', 37.5, False],
['uniform-l1', 50.0, False],
['uniform-l1', 75.0, False],
['uniform-l1', 100.0, False],
['uniform-l1', 125.0, False],
['uniform-l1', 150.0, False],
['uniform-l1', 175.0, False],
['uniform-l1', 200.0, False],
['uniform-l2', 0.25, False],
['uniform-l2', 0.5, False],
['uniform-l2', 0.75, False],
['uniform-l2', 1.0, False],
['uniform-l2', 1.5, False],
['uniform-l2', 2.0, False],
['uniform-l2', 2.5, False],
['uniform-l2', 3.0, False],
['uniform-l2', 4.0, False],
['uniform-l2', 5.0, False],
['uniform-l5', 0.03, False],
['uniform-l5', 0.06, False],
['uniform-l5', 0.1, False],
['uniform-l5', 0.15, False],
['uniform-l5', 0.2, False],
['uniform-l5', 0.25, False],
['uniform-l5', 0.3, False],
['uniform-l5', 0.4, False],
['uniform-l5', 0.5, False],
['uniform-l5', 0.6, False],
['uniform-l10', 0.02, False],
['uniform-l10', 0.03, False],
['uniform-l10', 0.05, False],
['uniform-l10', 0.07, False],
['uniform-l10', 0.1, False],
['uniform-l10', 0.13, False],
['uniform-l10', 0.16, False],
['uniform-l10', 0.2, False],
['uniform-l10', 0.25, False],
['uniform-l10', 0.3, False],
['uniform-l50', 0.01, False],
['uniform-l50', 0.02, False],
['uniform-l50', 0.03, False],
['uniform-l50', 0.04, False],
['uniform-l50', 0.06, False],
['uniform-l50', 0.08, False],
['uniform-l50', 0.1, False],
['uniform-l50', 0.12, False],
['uniform-l50', 0.15, False],
['uniform-l50', 0.18, False],
['uniform-l200', 0.01, False],
['uniform-l200', 0.02, False],
['uniform-l200', 0.03, False],
['uniform-l200', 0.04, False],
['uniform-l200', 0.05, False],
['uniform-l200', 0.07, False],
['uniform-l200', 0.09, False],
['uniform-l200', 0.11, False],
['uniform-l200', 0.13, False],
['uniform-l200', 0.15, False],
['uniform-l0-impulse', 0.005, True],
['uniform-l0-impulse', 0.01, True],
['uniform-l0-impulse', 0.015, True],
['uniform-l0-impulse', 0.02, True],
['uniform-l0-impulse', 0.03, True],
['uniform-l0-impulse', 0.04, True],
['uniform-l0-impulse', 0.06, True],
['uniform-l0-impulse', 0.08, True],
['uniform-l0-impulse', 0.1, True],
['uniform-l0-impulse', 0.12, True]
])
test_on_c = True
combine_test_corruptions = False #augment the test dataset with all corruptions
calculate_adv_distance = False
adv_distance_params = {'setsize': 1000, 'nb_iters': 100, 'eps_iter': 0.0005, 'norm': np.inf, "epsilon": 0.1}
calculate_autoattack_robustness = False
autoattack_params = {'setsize': 1000, 'epsilon': 8/255, 'norm': 'Linf'}

test_count = 2
if test_on_c:
    test_count += 22
if combine_test_corruptions:
    test_count += 1
else:
    test_count += test_corruptions.shape[0]
if calculate_adv_distance:
    test_count += 3
if calculate_autoattack_robustness:
    test_count += 2