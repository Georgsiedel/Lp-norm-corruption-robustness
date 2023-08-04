from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
from torch.utils.data import DataLoader
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from autoattack import AutoAttack

def pgd_with_early_stopping(model, inputs, labels, clean_predicted, eps, number_iterations, epsilon_iters, norm):

    for i in range(number_iterations):
        adv_inputs = projected_gradient_descent(model,
                                                inputs,
                                                eps=eps,
                                                eps_iter=epsilon_iters,
                                                nb_iter=1,
                                                norm=norm,
                                                y = labels,
                                                rand_init=False,
                                                sanity_checks=False)

        adv_outputs = model(adv_inputs)
        _, adv_predicted = torch.max(adv_outputs.data, 1)

        label_flipped = bool(adv_predicted!=clean_predicted)
        if label_flipped:
            break
        inputs = adv_inputs.clone()
    return adv_inputs, adv_predicted

def adv_distance(testloader, model, number_iterations, epsilon, eps_iter, norm, setsize):
    distance_list_0, image_idx_0 = [], []
    distance_list_1, image_idx_1 = [], []
    distance_list_2, image_idx_2 = [], []

    correct, total = 0, 0
    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        adv_inputs, adv_predicted = pgd_with_early_stopping(model, inputs, labels, predicted, epsilon, number_iterations, eps_iter, norm)

        distance = torch.norm((inputs - adv_inputs), p=norm)
        distance_list_0.append(distance) #all distances, also for originally misclassified points
        image_idx_0.append(i) #all points, also originally misclassified ones

        if (predicted == labels):
            distance = torch.norm((inputs - adv_inputs), p=norm)
            distance_list_1.append(distance)
            image_idx_1.append(i)
            distance_list_2.append(distance) #only originally correctly classified distances are counted
            image_idx_2.append(i) #only originally correctly classified points
        else:
            distance_list_1.append(0) #originally misclassified distances are counted as 0
            image_idx_1.append(i) #all points, also originally misclassified ones

        correct += (adv_predicted == labels).sum().item()
        total += labels.size(0)

        if (i+1) % 10 == 0:
            adv_acc = correct / total
            print(f"Completed: {i+1} of {setsize}, mean_distance: {sum(distance_list_0)/i}, correct: {correct}, total: {total}, accuracy: {adv_acc * 100}%")

    return distance_list_0, image_idx_0, distance_list_1, image_idx_1, distance_list_2, image_idx_2, adv_acc

def compute_adv_distance(testset, workers, model, adv_distance_params):
    truncated_testset, _ = torch.utils.data.random_split(testset,
                                                         [adv_distance_params["setsize"], len(testset)-adv_distance_params["setsize"]],
                                                         generator=torch.Generator().manual_seed(42))
    truncated_testloader = DataLoader(truncated_testset, batch_size=1, shuffle=False,
                                       pin_memory=True, num_workers=workers)
    epsilon = adv_distance_params["epsilon"]
    eps_iter = adv_distance_params["eps_iter"]
    nb_iters = adv_distance_params["nb_iters"]
    norm = adv_distance_params["norm"]
    dst0, idx0, dst1, idx1, dst2, idx2, adv_acc = adv_distance(testloader=truncated_testloader, model=model,
        number_iterations=nb_iters, epsilon=epsilon, eps_iter=eps_iter, norm=norm, setsize=adv_distance_params["setsize"])

    return adv_acc*100, dst0, idx0, dst1, idx1, dst2, idx2

def compute_adv_acc(autoattack_params, testset, model, workers, batchsize):
    truncated_testset, _ = torch.utils.data.random_split(testset, [autoattack_params["setsize"],
                                len(testset)-autoattack_params["setsize"]], generator=torch.Generator().manual_seed(1))
    truncated_testloader = DataLoader(truncated_testset, batch_size=autoattack_params["setsize"], shuffle=False,
                                       pin_memory=True, num_workers=workers)
    adversary = AutoAttack(model, norm=autoattack_params['norm'], eps=autoattack_params['epsilon'], version='standard')
    correct, total = 0, 0
    distance_list = []
    if autoattack_params["norm"] == 'Linf':
        autoattack_params["norm"] = np.inf
    for batch_id, (inputs, targets) in enumerate(truncated_testloader):
        adv_inputs, adv_predicted = adversary.run_standard_evaluation(inputs, targets, bs=50, return_labels=True)

        for i, (input) in enumerate(inputs):
            distance = torch.linalg.matrix_norm((input - adv_inputs[i]), ord=autoattack_params["norm"])
            distance_list.append(distance)

    mean_aa_dist = np.asarray(torch.tensor(distance_list).cpu()).mean()
    correct += (adv_predicted == targets).sum().item()
    total += targets.size(0)
    adv_acc = correct / total
    return adv_acc, mean_aa_dist