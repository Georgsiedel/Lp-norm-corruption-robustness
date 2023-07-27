from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# os.chdir('C:\\Users\\Admin\\Desktop\\Python\\corruption-testing')

if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    import importlib
    from experiments.eval import eval_metric
    import shutil
    import torch

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~5-15%
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1" #this blocks the spawn of multiple workers

    experiments_number = 28

    for experiment in [54, 58, 52, 51, 53, 55, 57]:#range(2, experiments_number):
        configname = (f'experiments.configs.config{experiment}')
        config = importlib.import_module(configname)

        print('Starting experiment number', experiment)
        runs = 1

        print('Beginning training on', config.dataset, 'dataset')
        for run in range(0, runs):
            torch.cuda.empty_cache()
            print("Training run #", run)
            if not config.combine_train_corruptions:
                for id, (noise_type, train_epsilon, max) in enumerate(config.train_corruptions):
                    print("Corruption training: ", noise_type, train_epsilon, 'and max-training:', max)
                    cmd0 = "python experiments/train.py --noise={} --epsilon={} --max={} --run={} --experiment={} " \
                           "--epochs={} --learningrate={} --dataset={} --validontest={} --lrschedule={} --lrparams=\"{}\" " \
                           "--earlystop={} --earlystopPatience={} --optimizer={} --optimizerparams=\"{}\" --modeltype={} " \
                           "--modelparams=\"{}\" --resize={} --aug_strat_check={} --train_aug_strat={} --jsd_loss={} " \
                           "--mixup_alpha={} --cutmix_alpha={} --combine_train_corruptions={} --concurrent_combinations={} " \
                           "--batchsize={} --number_workers={} --lossparams=\"{}\" --RandomEraseProbability={} " \
                           "--warmupepochs={} --normalize={}".format(noise_type, train_epsilon, max, run, experiment, config.epochs,
                                                   config.learningrate, config.dataset, config.validontest, config.lrschedule,
                                                   config.lrparams, config.earlystop, config.earlystopPatience,
                                                   config.optimizer, config.optimizerparams, config.modeltype,
                                                   config.modelparams, config.resize, config.aug_strat_check,
                                                   config.train_aug_strat, config.jsd_loss, config.mixup_alpha,
                                                   config.cutmix_alpha, config.combine_train_corruptions,
                                                   config.concurrent_combinations, config.batchsize, config.number_workers,
                                                   config.lossparams, config.RandomEraseProbability, config.warmupepochs,
                                                   config.normalize)
                    os.system(cmd0)

            if config.combine_train_corruptions:
                print('Combined training')
                cmd0 = "python experiments/train.py --run={} --experiment={} --epochs={} --learningrate={} --dataset={} " \
                       "--validontest={} --lrschedule={} --lrparams=\"{}\" --earlystop={} --earlystopPatience={} --optimizer={} " \
                       "--optimizerparams=\"{}\" --modeltype={} --modelparams=\"{}\" --resize={} --aug_strat_check={} " \
                       "--train_aug_strat={} --jsd_loss={} --mixup_alpha={} --cutmix_alpha={} --combine_train_corruptions={} " \
                       "--concurrent_combinations={} --batchsize={} --number_workers={} --lossparams=\"{}\" " \
                       "--RandomEraseProbability={} --warmupepochs={} --normalize={}"\
                    .format(run, experiment, config.epochs, config.learningrate, config.dataset, config.validontest,
                            config.lrschedule, config.lrparams, config.earlystop, config.earlystopPatience,
                            config.optimizer, config.optimizerparams, config.modeltype, config.modelparams, config.resize,
                            config.aug_strat_check, config.train_aug_strat, config.jsd_loss, config.mixup_alpha,
                            config.cutmix_alpha, config.combine_train_corruptions, config.concurrent_combinations,
                            config.batchsize, config.number_workers, config.lossparams, config.RandomEraseProbability,
                            config.warmupepochs, config.normalize)
                os.system(cmd0)

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
                filename = f'./experiments/models/{config.dataset}/{config.modeltype}/{config.lrschedule}/combined_training/{config.modeltype}_config{experiment}_concurrent_{config.concurrent_combinations}_run_{run}.pth'
                test_metric_col = eval_metric(filename, config.test_corruptions, config.combine_test_corruptions, config.test_on_c, config.modeltype, config.modelparams, config.resize, config.dataset, config.batchsize, config.number_workers)
                test_metrics[:, 0] = np.array(test_metric_col)
                print(test_metric_col)
            else:
                for idx, (noise_type, train_epsilon, max) in enumerate(config.train_corruptions):
                    print("Corruption training of type: ", noise_type, "with epsilon: ", train_epsilon, "and max-corruption =", max)
                    filename = f'./experiments/models/{config.dataset}/{config.modeltype}/{config.lrschedule}/separate_training/{config.modeltype}_{noise_type}_epsilon_{train_epsilon}_{max}_run_{run}.pth'
                    test_metric_col = eval_metric(filename, config.test_corruptions, config.combine_test_corruptions, config.test_on_c, config.modeltype, config.modelparams, config.resize, config.dataset, config.batchsize, config.number_workers)
                    test_metrics[:, idx] = np.array(test_metric_col)
                    print(test_metric_col)

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
            test_corruptions_string = np.loadtxt('./experiments/data/labels.txt', dtype=list)

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

        if config.combine_train_corruptions == True:
            avg_report_frame.to_csv(f'./results/{config.dataset}/{config.modeltype}/{config.lrschedule}/combined_training/'
                                    f'{config.modeltype}_config{experiment}_metrics_test_avg.csv', index=True, header=True,
                                    sep=';', float_format='%1.3f', decimal=',')
            max_report_frame.to_csv(f'./results/{config.dataset}/{config.modeltype}/{config.lrschedule}/combined_training/'
                                    f'{config.modeltype}_config{experiment}_metrics_test_max.csv', index=True, header=True,
                                    sep=';', float_format='%1.3f', decimal=',')
            std_report_frame.to_csv(f'./results/{config.dataset}/{config.modeltype}/{config.lrschedule}/combined_training/'
                                    f'{config.modeltype}_config{experiment}_metrics_test_std.csv', index=True, header=True,
                                    sep=';', float_format='%1.3f', decimal=',')
            shutil.copyfile(f'./experiments/configs/config{experiment}.py',
                            f'./results/{config.dataset}/{config.modeltype}/{config.lrschedule}/combined_training/config{experiment}.py')
        else:
            avg_report_frame.to_csv(f'./results/{config.dataset}/{config.modeltype}/{config.lrschedule}/separate_training/'
                                    f'{config.modeltype}_config{experiment}_metrics_test_avg.csv', index=True, header=True,
                                    sep=';', float_format='%1.3f', decimal=',')
            max_report_frame.to_csv(f'./results/{config.dataset}/{config.modeltype}/{config.lrschedule}/separate_training/'
                                    f'{config.modeltype}_config{experiment}_metrics_test_max.csv', index=True, header=True,
                                    sep=';', float_format='%1.3f', decimal=',')
            std_report_frame.to_csv(f'./results/{config.dataset}/{config.modeltype}/{config.lrschedule}/separate_training/'
                                    f'{config.modeltype}_config{experiment}_metrics_test_std.csv', index=True, header=True,
                                    sep=';', float_format='%1.3f', decimal=',')
            shutil.copyfile(f'./experiments/configs/config{experiment}.py',
                            f'./results/{config.dataset}/{config.modeltype}/{config.lrschedule}/separate_training/config{experiment}.py')