import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~5-15%
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" #this blocks the spawn of multiple workers

if __name__ == '__main__':
    import numpy as np
    import importlib
    from experiments.eval import eval_metric
    from experiments.visuals_and_reports import create_report

    for experiment in list(range(213,215)) + [216,217] + [215,218,219] + [215,218,219] + [215,218,219]:

        configname = (f'experiments.configs.config{experiment}')
        config = importlib.import_module(configname)

        print('Starting experiment #',experiment, 'on', config.dataset, 'dataset')
        runs = 5

        for run in range(runs):
            print("Training run #",run)

            if experiment in list(range(212,220)):
                resume = True
            else:
                resume = False

            if not config.combine_train_corruptions:
                for id, traincorruption in enumerate(config.train_corruptions):
                    print("Separate corruption training of type: ", traincorruption)
                    cmd0 = "python experiments/train.py --resume={} --id={} --run={} --experiment={} " \
                           "--epochs={} --learningrate={} --dataset={} --validontest={} --lrschedule={} --lrparams=\"{}\" " \
                           "--earlystop={} --earlystopPatience={} --optimizer={} --optimizerparams=\"{}\" --modeltype={} " \
                           "--modelparams=\"{}\" --resize={} --aug_strat_check={} --train_aug_strat={} --jsd_loss={} " \
                           "--mixup_alpha={} --cutmix_alpha={} --combine_train_corruptions={} --concurrent_combinations={} " \
                           "--batchsize={} --number_workers={} --lossparams=\"{}\" --RandomEraseProbability={} " \
                           "--warmupepochs={} --normalize={} --num_classes={} --pixel_factor={} --noise_patch_scale=\"{}\" " \
                           "--random_noise_dist={} "\
                        .format(resume, id, run, experiment, config.epochs, config.learningrate,
                                config.dataset, config.validontest, config.lrschedule, config.lrparams, config.earlystop,
                                config.earlystopPatience, config.optimizer, config.optimizerparams, config.modeltype,
                                config.modelparams, config.resize, config.aug_strat_check, config.train_aug_strat,
                                config.jsd_loss, config.mixup_alpha, config.cutmix_alpha, config.combine_train_corruptions,
                                config.concurrent_combinations, config.batchsize, config.number_workers, config.lossparams,
                                config.RandomEraseProbability, config.warmupepochs, config.normalize, config.num_classes,
                                config.pixel_factor, config.noise_patch_scale, config.random_noise_dist)
                    if experiment in []:
                        print('skip training')
                    else:
                        os.system(cmd0)

            if config.combine_train_corruptions:
                print('Combined training')
                cmd0 = "python experiments/train.py --resume={} --run={} --experiment={} --epochs={} --learningrate={} --dataset={} " \
                       "--validontest={} --lrschedule={} --lrparams=\"{}\" --earlystop={} --earlystopPatience={} --optimizer={} " \
                       "--optimizerparams=\"{}\" --modeltype={} --modelparams=\"{}\" --resize={} --aug_strat_check={} " \
                       "--train_aug_strat={} --jsd_loss={} --mixup_alpha={} --cutmix_alpha={} --combine_train_corruptions={} " \
                       "--concurrent_combinations={} --batchsize={} --number_workers={} --lossparams=\"{}\" " \
                       "--RandomEraseProbability={} --warmupepochs={} --normalize={} --num_classes={} --pixel_factor={} " \
                       "--noise_patch_scale=\"{}\" --random_noise_dist={} "\
                    .format(resume, run, experiment, config.epochs, config.learningrate, config.dataset, config.validontest,
                            config.lrschedule, config.lrparams, config.earlystop, config.earlystopPatience,
                            config.optimizer, config.optimizerparams, config.modeltype, config.modelparams, config.resize,
                            config.aug_strat_check, config.train_aug_strat, config.jsd_loss, config.mixup_alpha,
                            config.cutmix_alpha, config.combine_train_corruptions, config.concurrent_combinations,
                            config.batchsize, config.number_workers, config.lossparams, config.RandomEraseProbability,
                            config.warmupepochs, config.normalize, config.num_classes, config.pixel_factor,
                            config.noise_patch_scale, config.random_noise_dist)
                if experiment in []:
                    print('skip training')
                else:
                    os.system(cmd0)

        if experiment in list(range(212,215)) + [216,217]+ [215,218,219]:#
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
                                                  config.modeltype, config.modelparams, config.resize, config.dataset, 1024,
                                                  0, config.normalize, config.calculate_adv_distance, config.adv_distance_params,
                                                  config.calculate_autoattack_robustness, config.autoattack_params, config.pixel_factor)
                    test_metrics[:, 0] = np.array(test_metric_col)
                    print(test_metric_col)
                else:
                    for idx, traincorruption in enumerate(config.train_corruptions):
                        training_folder = 'combined' if config.combine_train_corruptions == True else 'separate'
                        if isinstance(traincorruption[1], dict):
                            string = ','.join([f"{k}={v}" for k,v in traincorruption[1].items()])
                        else:
                            string = traincorruption[1]
                        filename_spec = str(f"_{traincorruption[0]}_{string}_" if
                                            config.combine_train_corruptions == False else f"_")
                        print("Evaluating model trained with corruption of type: ", traincorruption)
                        filename = f'./experiments/trained_models/{config.dataset}/{config.modeltype}/config' \
                                   f'{experiment}_{config.lrschedule}_{training_folder}{filename_spec}run_{run}.pth'
                        test_metric_col = eval_metric(filename, config.test_corruptions, config.combine_test_corruptions, config.test_on_c,
                                                      config.modeltype, config.modelparams, config.resize, config.dataset, 1000,
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