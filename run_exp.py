# Training and Evaluation Code and Network architecture inspired/adopted from https://github.com/wangben88/statistically-robust-nn-classification
# and https://github.com/yangarbiter/robust-local-lipschitz

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.chdir('C:\\Users\\Admin\\Desktop\\Python\\corruption-testing')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from experiments.eval import eval_metric
from experiments.config import train_corruptions
from experiments.config import test_corruptions
from experiments.config import combine_train_corruptions
from experiments.config import combine_test_corruptions
from experiments.config import model_count
from experiments.config import test_count
from experiments.config import test_on_c
from experiments.config import concurrent_combinations

runs = 1

# Train network on CIFAR-10 for natural training, two versions of corruption training, and PGD adversarial training.
# Progressively smaller learning rates are used over training
print('Beginning training of Wide ResNet networks on CIFAR-10')
#for run in range(4, runs):
for run in range(0, runs):
    print("Training run #", run)
    if not combine_train_corruptions:
        #for id, (noise_type, train_epsilon, max) in enumerate(train_corruptions[3:15]):
        for id, (noise_type, train_epsilon, max) in enumerate(train_corruptions):
            print("Corruption training: ", noise_type, train_epsilon)
            cmd0 = 'python experiments/train.py --noise={} --epsilon={} --epochs=85 --lr=0.01 --run={} --max={}'.format(
                noise_type, train_epsilon, run, max)
            cmd1 = 'python experiments/train.py --resume --noise={} --epsilon={} --epochs=10 --lr=0.002 --run={} --max={}'.format(
                noise_type, train_epsilon, run, max)
            cmd2 = 'python experiments/train.py --resume --noise={} --epsilon={} --epochs=5 --lr=0.0004 --run={} --max={}'.format(
                noise_type, train_epsilon, run, max)
            os.system(cmd0)
            os.system(cmd1)
            os.system(cmd2)
    if combine_train_corruptions:
        print('Combined training')
        cmd0 = 'python experiments/train.py --epochs=85 --lr=0.01 --run={}'.format(
            run)
        cmd1 = 'python experiments/train.py --resume --epochs=10 --lr=0.002 --run={}'.format(
            run)
        cmd2 = 'python experiments/train.py --resume --epochs=5 --lr=0.0004 --run={}'.format(
            run)
        os.system(cmd0)
        os.system(cmd1)
        os.system(cmd2)

# Calculate accuracy and robust accuracy, evaluating each trained network on each corruption
print('Beginning metric evaluation')
# Evaluation on train/test set respectively
all_test_metrics = np.empty([test_count, model_count, runs])
#all_mscr = np.empty([model_count, runs])
#std_mscr = np.empty(model_count)
#all_dif = np.empty([4, runs])
#all_avg = np.empty([4])
#all_std = np.empty([4])
avg_test_metrics = np.empty([test_count, model_count])
std_test_metrics = np.empty([test_count, model_count])
max_test_metrics = np.empty([test_count, model_count])

#acc1 = np.empty(runs)
#acc_series1 = np.empty(runs)

for run in range(runs):
    print("Metric evaluation for training run #", run)
    test_metrics = np.empty([test_count, model_count])

    if combine_train_corruptions:
        print("Corruption training of combined type")
        filename = f'./experiments/models/cifar_combined_0_concurrent_{concurrent_combinations}_run_{run}.pth'
        test_metric_col = eval_metric(filename, test_corruptions, combine_test_corruptions, test_on_c)
        test_metrics[:, 0] = np.array(test_metric_col)
        print(test_metric_col)
    else:
        for idx, (noise_type, train_epsilon, max) in enumerate(train_corruptions):
            print("Corruption training of type: ", noise_type, "with epsilon: ", train_epsilon, "and max-corruption =", max)
            filename = './experiments/models/{}/cifar_epsilon_{}_run_{}.pth'.format(noise_type, train_epsilon, run)
            test_metric_col = eval_metric(filename, test_corruptions, combine_test_corruptions, test_on_c)
            test_metrics[:, idx] = np.array(test_metric_col)
            #mscr: calculated from clean and r-separated testing sets
            #all_mscr[idx, run] = (np.array(test_metric_col)[6] - np.array(test_metric_col)[0]) / np.array(test_metric_col)[0]

    all_test_metrics[:test_count, :model_count, run] = test_metrics
    #all_dif[0, run] = all_test_metrics[0, 1, run] - all_test_metrics[0, 0, run]
    #all_dif[1, run] = all_test_metrics[0, 2, run] - all_test_metrics[0, 0, run]
    #all_dif[2, run] = all_test_metrics[0, 3, run] - all_test_metrics[0, 0, run]
    #all_dif[3, run] = all_test_metrics[0, 4, run] - all_test_metrics[0, 0, run]

#    acc1[run] = test_metrics[0, 0]
#    acc_series1[run] = acc1[:run+1].mean()
#
#    np.savetxt(
#        './results/cifar10_metrics_test_run_{}.csv'.format(
#        run), test_metrics, fmt='%1.3f', delimiter=';', header='Networks trained with'
#        ' corruptions (epsilon = {}) along columns THEN evaluated on test set using A-TRSM (epsilon = {}) '
#        ' along rows'.format(model_epsilons_str, eval_epsilons_str, ))

for idm in range(model_count):
    #std_mscr[idm] = all_mscr[idm, :].std()
    for ide in range(test_count):
        avg_test_metrics[ide, idm] = all_test_metrics[ide, idm, :].mean()
        std_test_metrics[ide, idm] = all_test_metrics[ide, idm, :].std()
        max_test_metrics[ide, idm] = all_test_metrics[ide, idm, :].max()

#all_std[0] = all_dif[0,:].std()
#all_avg[0] = all_dif[0,:].mean()
#all_std[1] = all_dif[1,:].std()
#all_avg[1] = all_dif[1,:].mean()
#all_std[2] = all_dif[2,:].std()
#all_avg[2] = all_dif[2,:].mean()
#all_std[3] = all_dif[3,:].std()
#all_avg[3] = all_dif[3,:].mean()
#print(all_avg)
#print(all_std)
#print(acc1)
#print(acc1.std())
#x1 = list(range(1, runs + 1))
#y1 = acc_series1
#plt.scatter(x1, y1)
#plt.xlabel("Runs")
#plt.ylabel("Average Test Acc")
#plt.title("Convergence of average Test Accuracy over runs")
#plt.legend(["train&test e = 0"], loc='right')
#plt.show()
test_corruptions_string = np.empty([test_count])
if combine_train_corruptions == True:
    train_corruptions_string = ['config']
else:
    train_corruptions_string = train_corruptions.astype(str)
    train_corruptions_string = np.array([','.join(row) for row in train_corruptions_string])

if test_on_c == True:
    test_corruptions_string = np.loadtxt('./experiments/data/cifar-10-c/labels.txt', dtype=list)

if combine_test_corruptions == True:
    test_corruptions_label = ['config']
    test_corruptions_string = np.append(test_corruptions_string, test_corruptions_label)
else:
    test_corruptions_labels = test_corruptions.astype(str)
    test_corruptions_labels = np.array([','.join(row) for row in test_corruptions_labels])
    test_corruptions_string = np.append(test_corruptions_string, test_corruptions_labels)

avg_report_frame = pd.DataFrame(avg_test_metrics, index=test_corruptions_string, columns=train_corruptions_string)
max_report_frame = pd.DataFrame(max_test_metrics, index=test_corruptions_string, columns=train_corruptions_string)
std_report_frame = pd.DataFrame(std_test_metrics, index=test_corruptions_string, columns=train_corruptions_string)

avg_report_frame.to_csv('./results/cifar10_metrics_test_avg.csv', index=True, header=True, sep=';')
max_report_frame.to_csv('./results/cifar10_metrics_test_max.csv', index=True, header=True, sep=';')
std_report_frame.to_csv('./results/cifar10_metrics_test_std.csv', index=True, header=True, sep=';')

#np.savetxt(
#    './results/cifar10_metrics_test_avg.csv',
#    avg_report_frame, fmt='%1.3f', delimiter=';', header='Networks trained with'
#    ' corruptions (epsilon = {}) along columns THEN evaluated on test set with corruptions (epsilon = {}) '
#    ' along rows'.format(model_epsilons_str, eval_epsilons_str))
#np.savetxt(
#    './results/cifar10_metrics_test_max.csv',
#    max_report_frame, fmt='%1.3f', delimiter=';', header='Networks trained with'
#    ' corruptions (epsilon = {}) along columns THEN evaluated on test set with corruptions (epsilon = {}) '
#    ' along rows'.format(model_epsilons_str, eval_epsilons_str))
#np.savetxt(
#    './results/diff.csv',
#    all_dif, fmt='%1.3f', delimiter=';')
#np.savetxt(
#    './results/cifar10_metrics_test_std.csv',
#    std_report_frame, fmt='%1.4f', delimiter=';', header='Networks trained with'
#    ' corruptions (epsilon = {}) along columns THEN evaluated on test set with corruptions (epsilon = {}) '
#    ' along rows'.format(model_epsilons_str, eval_epsilons_str))
#np.savetxt(
#    './results/cifar10_mscr_std.csv',
#    std_mscr, fmt='%1.4f', delimiter=';', header='Networks trained with'
#    ' corruptions (epsilon = {}) along columns'.format(model_epsilons_str))
