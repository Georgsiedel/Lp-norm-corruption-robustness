from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.chdir('C:\\Users\\Admin\\Desktop\\Python\\corruption-testing')
import numpy as np
import pandas as pd
import importlib
from experiments.eval import eval_metric

experiments_number = 10

for experiment in [0]: #range(experiments_number):
    configname = (f'experiments.configs.config{experiment}')
    config = importlib.import_module(configname)

    print('Starting experiment number', experiment)
    runs = 5

    # Train network on CIFAR-10 for natural training, two versions of corruption training, and PGD adversarial training.
    # Progressively smaller learning rates are used over training

    print('Beginning training on CIFAR-10')
    for run in range(0, runs):
        print("Training run #", run)
        if not config.combine_train_corruptions:
            for id, (noise_type, train_epsilon, max) in enumerate(config.train_corruptions):
                print("Corruption training: ", noise_type, train_epsilon)
                cmd0 = 'python experiments/train.py --noise={} --epsilon={} --epochs=85 --lr=0.01 --run={} --max={} --experiment={}'.format(
                    noise_type, train_epsilon, run, max, experiment)
                cmd1 = 'python experiments/train.py --resume --noise={} --epsilon={} --epochs=10 --lr=0.002 --run={} --max={} --experiment={}'.format(
                    noise_type, train_epsilon, run, max, experiment)
                cmd2 = 'python experiments/train.py --resume --noise={} --epsilon={} --epochs=5 --lr=0.0004 --run={} --max={} --experiment={}'.format(
                    noise_type, train_epsilon, run, max, experiment)
                os.system(cmd0)
                os.system(cmd1)
                os.system(cmd2)
        if config.combine_train_corruptions:
            print('Combined training')
            cmd0 = 'python experiments/train.py --epochs=85 --lr=0.01 --run={} --experiment={}'.format(
                run, experiment)
            cmd1 = 'python experiments/train.py --resume --epochs=10 --lr=0.002 --run={} --experiment={}'.format(
                run, experiment)
            cmd2 = 'python experiments/train.py --resume --epochs=5 --lr=0.0004 --run={} --experiment={}'.format(
                run, experiment)
            os.system(cmd0)
            os.system(cmd1)
            os.system(cmd2)

    # Calculate accuracy and robust accuracy, evaluating each trained network on each corruption
    print('Beginning metric evaluation')
    # Evaluation on train/test set respectively
    all_test_metrics = np.empty([config.test_count, config.model_count, runs])

    avg_test_metrics = np.empty([config.test_count, config.model_count])
    std_test_metrics = np.empty([config.test_count, config.model_count])
    max_test_metrics = np.empty([config.test_count, config.model_count])

    for run in range(runs):
        print("Metric evaluation for training run #", run)
        test_metrics = np.empty([config.test_count, config.model_count])

        if config.combine_train_corruptions:
            print("Corruption training of combined type")
            filename = f'./experiments/models/{config.modeltype}_config{experiment}_concurrent_{config.concurrent_combinations}_run_{run}.pth'
            test_metric_col = eval_metric(filename, config.test_corruptions, config.combine_test_corruptions, config.test_on_c, config.modeltype)
            test_metrics[:, 0] = np.array(test_metric_col)
            print(test_metric_col)
        else:
            for idx, (noise_type, train_epsilon, max) in enumerate(config.train_corruptions):
                print("Corruption training of type: ", noise_type, "with epsilon: ", train_epsilon, "and max-corruption =", max)
                filename = './experiments/models/{}/{}_epsilon_{}_run_{}.pth'.format(noise_type, config.modeltype, train_epsilon, run)
                test_metric_col = eval_metric(filename, config.test_corruptions, config.combine_test_corruptions, config.test_on_c, config.modeltype)
                test_metrics[:, idx] = np.array(test_metric_col)

        all_test_metrics[:config.test_count, :config.model_count, run] = test_metrics

    for idm in range(config.model_count):
        #std_mscr[idm] = all_mscr[idm, :].std()
        for ide in range(config.test_count):
            avg_test_metrics[ide, idm] = all_test_metrics[ide, idm, :].mean()
            std_test_metrics[ide, idm] = all_test_metrics[ide, idm, :].std()
            max_test_metrics[ide, idm] = all_test_metrics[ide, idm, :].max()

    test_corruptions_string = np.empty([config.test_count])
    if config.combine_train_corruptions == True:
        train_corruptions_string = ['config']
    else:
        train_corruptions_string = config.train_corruptions.astype(str)
        train_corruptions_string = np.array([','.join(row) for row in train_corruptions_string])

    if config.test_on_c == True:
        test_corruptions_string = np.loadtxt('./experiments/data/cifar-10-c/labels.txt', dtype=list)

    if config.combine_test_corruptions == True:
        test_corruptions_label = ['config']
        test_corruptions_string = np.append(test_corruptions_string, test_corruptions_label)
    else:
        test_corruptions_labels = config.test_corruptions.astype(str)
        test_corruptions_labels = np.array([','.join(row) for row in test_corruptions_labels])
        test_corruptions_string = np.append(test_corruptions_string, test_corruptions_labels)

    avg_report_frame = pd.DataFrame(avg_test_metrics, index=test_corruptions_string, columns=train_corruptions_string)
    max_report_frame = pd.DataFrame(max_test_metrics, index=test_corruptions_string, columns=train_corruptions_string)
    std_report_frame = pd.DataFrame(std_test_metrics, index=test_corruptions_string, columns=train_corruptions_string)

    avg_report_frame.to_csv(f'./results/{config.modeltype}_config{experiment}_metrics_test_avg.csv', index=True, header=True, sep=';', float_format='%1.3f', decimal=',')
    max_report_frame.to_csv(f'./results/{config.modeltype}_config{experiment}_metrics_test_max.csv', index=True, header=True, sep=';', float_format='%1.3f', decimal=',')
    std_report_frame.to_csv(f'./results/{config.modeltype}_config{experiment}_metrics_test_std.csv', index=True, header=True, sep=';', float_format='%1.3f', decimal=',')