import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_report(avg_test_metrics, max_test_metrics, std_test_metrics, train_corruptions, test_corruptions,
                combine_train_corruptions, combine_test_corruptions, dataset, modeltype, lrschedule, experiment,
                  test_on_c, calculate_adv_distance, calculate_autoattack_robustness, runs):

    training_folder = 'combined' if combine_train_corruptions == True else 'separate'

    if combine_train_corruptions == True:
        train_corruptions_string = ['config_model']
    else:
        train_corruptions_string = train_corruptions.astype(str)
        train_corruptions_string = np.array([','.join(row) for row in train_corruptions_string])

    test_corruptions_string = np.array(['Standard_Acc', 'RMSCE'])
    if test_on_c == True:
        test_corruptions_label = np.loadtxt('./experiments/data/c-labels.txt', dtype=list)
        test_corruptions_string = np.append(test_corruptions_string, test_corruptions_label, axis=0)
        test_corruptions_string = np.append(test_corruptions_string, ['RMSCE_C'], axis=0)
    if combine_test_corruptions == True:
        test_corruptions_label = ['Acc_config']
        test_corruptions_string = np.append(test_corruptions_string, test_corruptions_label)
    else:
        test_corruptions_labels = test_corruptions.astype(str)
        test_corruptions_labels = np.array([','.join(row) for row in test_corruptions_labels])
        test_corruptions_string = np.append(test_corruptions_string, test_corruptions_labels)

    if calculate_adv_distance == True:
        test_corruptions_string = np.append(test_corruptions_string, ['Acc_from_PGD_adv_distance_calculation',
                                                                      'Mean_adv_distance_with_misclassified_images_0)',
                                                                      'Mean_adv_distance_misclassified-images_not_included)'],
                                            axis=0)
    if calculate_autoattack_robustness == True:
        test_corruptions_string = np.append(test_corruptions_string,
                                            ['Adversarial_accuracy_autoattack', 'Mean_adv_distance_autoattack)'],
                                            axis=0)

    avg_report_frame = pd.DataFrame(avg_test_metrics, index=test_corruptions_string, columns=train_corruptions_string)
    avg_report_frame.to_csv(f'./results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}_'
                            f'metrics_test_avg.csv', index=True, header=True,
                            sep=';', float_format='%1.4f', decimal=',')
    if runs >= 2:
        max_report_frame = pd.DataFrame(max_test_metrics, index=test_corruptions_string, columns=train_corruptions_string)
        std_report_frame = pd.DataFrame(std_test_metrics, index=test_corruptions_string, columns=train_corruptions_string)
        max_report_frame.to_csv(f'./results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}_'
                                f'metrics_test_max.csv', index=True, header=True,
                                sep=';', float_format='%1.4f', decimal=',')
        std_report_frame.to_csv(f'./results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}_'
                                f'metrics_test_std.csv', index=True, header=True,
                                sep=';', float_format='%1.4f', decimal=',')

def learning_curves(dataset, modeltype, lrschedule, experiment, run, train_accs, valid_accs,
                    train_losses, valid_losses, training_folder, filename_spec):

    learning_curve_frame = pd.DataFrame({"train_accuracy": train_accs, "train_loss": train_losses,
                                         "valid_accuracy": valid_accs, "valid_loss": valid_losses})

    x = list(range(1, len(train_accs) + 1))
    plt.plot(x, train_accs, label='Train Accuracy')
    plt.plot(x, valid_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(np.linspace(1, len(train_accs), num=10, dtype=int))
    plt.legend(loc='best')

    learning_curve_frame.to_csv(f'./results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}'
                                f'_learning_curve{filename_spec}run_{run}.csv',
                                index=False, header=True, sep=';', float_format='%1.4f', decimal=',')
    plt.savefig(f'results/{dataset}/{modeltype}/config{experiment}_{lrschedule}_{training_folder}_learning_curve'
                f'{filename_spec}run_{run}.svg')

