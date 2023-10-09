import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if __name__ == '__main__':
    import numpy as np
    import importlib
    from experiments.eval import eval_metric
    from experiments.visuals_and_reports import create_report

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~5-15%
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1" #this blocks the spawn of multiple workers

    for experiment in list([99,100,101,79,80,81,84,85,87,88,90,91,92,93,94,97,98,103,104,105,106,107,110,111,113,114,116,117]) + list([1,2,3,6,7,9,10,12,13,14,15,16,19,20,22,23,25,26,27,28,29,32,33,35,36,38,39,40,41,42,45,46,48,49,51,52,53,54,55,58,59,61,62,64,65,66,67,68,71,72,74,75,77,78]):#range(2, experiments_number):

        configname = (f'experiments.configs.config{experiment}')
        config = importlib.import_module(configname)

        print('Starting experiment #',experiment, 'on', config.dataset, 'dataset')
        runs = 1
        resume = False


        for run in range(runs):
            print("Training run #",run)
            if not config.combine_train_corruptions:
                for id, (noise_type, train_epsilon, max) in enumerate(config.train_corruptions):
                    if experiment == 99 and id==0:
                        resume = True
                    else:
                        resume=False
                    print("Separate corruption training: ", noise_type, train_epsilon, 'and max-training:', max)
                    cmd0 = "python experiments/train.py --resume={} --noise={} --epsilon={} --max={} --run={} --experiment={} " \
                           "--epochs={} --learningrate={} --dataset={} --validontest={} --lrschedule={} --lrparams=\"{}\" " \
                           "--earlystop={} --earlystopPatience={} --optimizer={} --optimizerparams=\"{}\" --modeltype={} " \
                           "--modelparams=\"{}\" --resize={} --aug_strat_check={} --train_aug_strat={} --jsd_loss={} " \
                           "--mixup_alpha={} --cutmix_alpha={} --combine_train_corruptions={} --concurrent_combinations={} " \
                           "--batchsize={} --number_workers={} --lossparams=\"{}\" --RandomEraseProbability={} " \
                           "--warmupepochs={} --normalize={} --num_classes={} --pixel_factor={}"\
                        .format(resume, noise_type, train_epsilon, max, run, experiment, config.epochs, config.learningrate,
                                config.dataset, config.validontest, config.lrschedule, config.lrparams, config.earlystop,
                                config.earlystopPatience, config.optimizer, config.optimizerparams, config.modeltype,
                                config.modelparams, config.resize, config.aug_strat_check, config.train_aug_strat,
                                config.jsd_loss, config.mixup_alpha, config.cutmix_alpha, config.combine_train_corruptions,
                                config.concurrent_combinations, config.batchsize, config.number_workers, config.lossparams,
                                config.RandomEraseProbability, config.warmupepochs, config.normalize, config.num_classes,
                                config.pixel_factor)
                    if experiment > 92 or id > 8:
                        os.system(cmd0)

            if config.combine_train_corruptions:
                print('Combined training')
                cmd0 = "python experiments/train.py --resume={} --run={} --experiment={} --epochs={} --learningrate={} --dataset={} " \
                       "--validontest={} --lrschedule={} --lrparams=\"{}\" --earlystop={} --earlystopPatience={} --optimizer={} " \
                       "--optimizerparams=\"{}\" --modeltype={} --modelparams=\"{}\" --resize={} --aug_strat_check={} " \
                       "--train_aug_strat={} --jsd_loss={} --mixup_alpha={} --cutmix_alpha={} --combine_train_corruptions={} " \
                       "--concurrent_combinations={} --batchsize={} --number_workers={} --lossparams=\"{}\" " \
                       "--RandomEraseProbability={} --warmupepochs={} --normalize={} --num_classes={} --pixel_factor={}"\
                    .format(resume, run, experiment, config.epochs, config.learningrate, config.dataset, config.validontest,
                            config.lrschedule, config.lrparams, config.earlystop, config.earlystopPatience,
                            config.optimizer, config.optimizerparams, config.modeltype, config.modelparams, config.resize,
                            config.aug_strat_check, config.train_aug_strat, config.jsd_loss, config.mixup_alpha,
                            config.cutmix_alpha, config.combine_train_corruptions, config.concurrent_combinations,
                            config.batchsize, config.number_workers, config.lossparams, config.RandomEraseProbability,
                            config.warmupepochs, config.normalize, config.num_classes, config.pixel_factor)
                os.system(cmd0)

        # Calculate accuracy and robust accuracy, evaluating each trained network on each corruption
        print('Beginning metric evaluation')
        all_test_metrics = np.empty([config.test_count, config.model_count, runs])
        avg_test_metrics = np.empty([config.test_count, config.model_count])
        std_test_metrics = np.empty([config.test_count, config.model_count])
        max_test_metrics = np.empty([config.test_count, config.model_count])

        for run in range(runs):
            print("Evaluation run #",run)
            test_metrics = np.empty([config.test_count, config.model_count])

            if config.combine_train_corruptions:
                print("Evaluating model of combined type")
                filename = f'./experiments/trained_models/{config.dataset}/{config.modeltype}/config{experiment}_' \
                           f'{config.lrschedule}_combined_run_{run}.pth'
                test_metric_col = eval_metric(filename, config.test_corruptions, config.combine_test_corruptions, config.test_on_c,
                                              config.modeltype, config.modelparams, config.resize, config.dataset, 2048,
                                              config.number_workers, config.normalize, config.calculate_adv_distance, config.adv_distance_params,
                                              config.calculate_autoattack_robustness, config.autoattack_params, config.pixel_factor)
                test_metrics[:, 0] = np.array(test_metric_col)
                print(test_metric_col)
            else:
                for idx, (noise_type, train_epsilon, max) in enumerate(config.train_corruptions):
                    print("Evaluating model trained on corruption of type: ", noise_type, "with epsilon: ", train_epsilon, "and max-corruption =", max)
                    filename = f'./experiments/trained_models/{config.dataset}/{config.modeltype}/config{experiment}_' \
                               f'{config.lrschedule}_separate_{noise_type}_eps_{train_epsilon}_{max}_run_{run}.pth'
                    test_metric_col = eval_metric(filename, config.test_corruptions, config.combine_test_corruptions, config.test_on_c,
                                                  config.modeltype, config.modelparams, config.resize, config.dataset, 2048,
                                                  config.number_workers, config.normalize, config.calculate_adv_distance, config.adv_distance_params,
                                                  config.calculate_autoattack_robustness, config.autoattack_params, config.pixel_factor)
                    test_metrics[:, idx] = np.array(test_metric_col)
                    print(test_metric_col)

            all_test_metrics[:config.test_count, :config.model_count, run] = test_metrics

        for idm in range(config.model_count):
            for ide in range(config.test_count):
                avg_test_metrics[ide, idm] = all_test_metrics[ide, idm, :].mean()
                std_test_metrics[ide, idm] = all_test_metrics[ide, idm, :].std()
                max_test_metrics[ide, idm] = all_test_metrics[ide, idm, :].max()

        create_report(avg_test_metrics, max_test_metrics, std_test_metrics, config.train_corruptions, config.test_corruptions,
                      config.combine_train_corruptions, config.combine_test_corruptions, config.dataset, config.modeltype,
                      config.lrschedule, experiment, config.test_on_c, config.calculate_adv_distance,
                      config.calculate_autoattack_robustness, runs)