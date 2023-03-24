import numpy as np

train_corruptions = np.array([
['standard', 0.0, False],
['uniform-linf', 0.02, False],
['uniform-linf', 0.03, False],
#['uniform-linf', 0.05, False],
#['uniform-linf', 0.1, False],
#['uniform-linf-brightness', 0.05, False],
#['uniform-linf-brightness', 0.2, False],
['uniform-l0.5', 100000.0, False],
['uniform-l0.5', 200000.0, False],
#['uniform-l0.5', 500000.0, False],
['uniform-l1', 40.0, False],
['uniform-l1', 80.0, False],
#['uniform-l1', 200.0, False],
['uniform-l2', 1.0, False],
['uniform-l2', 2.0, False],
#['uniform-l2', 4.0, False],
#['uniform-l5', 0.2, False],
#['uniform-l5', 0.4, False],
#['uniform-l5', 1.0, False],
#['uniform-l10', 0.15, False],
#['uniform-l10', 0.3, False],
#['uniform-l10', 0.7, False],
['uniform-l50', 0.1, False],
['uniform-l50', 0.2, False],
#['uniform-l50', 0.5, False],
#['uniform-l200', 0.1, False],
#['uniform-l200', 0.2, False],
#['uniform-l200', 0.5, False],
#['uniform-l0-salt-pepper', 0.01, True],
#['uniform-l0-salt-pepper', 0.02, True],
['uniform-l0-impulse-max', 0.01, True],
['uniform-l0-impulse-max', 0.02, True],
#['uniform-l0-impulse-max', 0.04, True],
['uniform-l0-impulse-linear', 0.02, False],
['uniform-l0-impulse-linear', 0.03, False],
['uniform-l0-impulse-linear', 0.06, False]
])
combine_train_corruptions = False #augment the train dataset with all corruptions

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
['uniform-linf', 0.02, False],
['uniform-linf', 0.03, False],
['uniform-linf', 0.05, False],
['uniform-linf', 0.1, False],
['uniform-linf-brightness', 0.05, False],
['uniform-linf-brightness', 0.2, False],
['uniform-l0.5', 100000.0, False],
['uniform-l0.5', 200000.0, False],
['uniform-l0.5', 500000.0, False],
['uniform-l1', 40.0, False],
['uniform-l1', 80.0, False],
['uniform-l1', 200.0, False],
['uniform-l2', 1.0, False],
['uniform-l2', 2.0, False],
['uniform-l2', 4.0, False],
['uniform-l5', 0.2, False],
['uniform-l5', 0.4, False],
['uniform-l5', 1.0, False],
['uniform-l10', 0.15, False],
['uniform-l10', 0.3, False],
['uniform-l10', 0.7, False],
['uniform-l50', 0.1, False],
['uniform-l50', 0.2, False],
['uniform-l50', 0.5, False],
['uniform-l200', 0.1, False],
['uniform-l200', 0.2, False],
['uniform-l200', 0.5, False],
['uniform-l0-salt-pepper', 0.01, True],
['uniform-l0-salt-pepper', 0.02, True],
['uniform-l0-impulse-max', 0.01, True],
['uniform-l0-impulse-max', 0.02, True],
['uniform-l0-impulse-max', 0.04, True],
['uniform-l0-impulse-linear', 0.02, False],
['uniform-l0-impulse-linear', 0.03, False],
['uniform-l0-impulse-linear', 0.06, False]
])
test_on_c = True
combine_test_corruptions = False #augment the test dataset with all corruptions
clean_validation = True
if test_on_c:
    if combine_test_corruptions:
        test_count = 1 + 20
    else:
        test_count = test_corruptions.shape[0] + 20
else:
    if combine_test_corruptions:
        test_count = 1
    else:
        test_count = test_corruptions.shape[0]

if __name__ == '__main__':
    #show corrupted images of certain configuration
    #use this to check whether corruptions are invisible or they make classes imperceptible
    from sample_corrupted_img import sample_corr_img
    import matplotlib.pyplot as plt
    fig = sample_corr_img(3, False, 'uniform-l0-impulse-max', 0.01, 'no')
    plt.show()

    #calculate minimal distance of points from different classes
    #from distance import get_nearest_oppo_dist
    #import pandas as pd
    #dist = np.inf #0.2, ..., 1, 2, ...,  np.inf
    #traintrain_ret, traintest_ret, testtest_ret = get_nearest_oppo_dist(dist)
    #ret = np.array([[traintrain_ret.min(), traintest_ret.min(), testtest_ret.min()], [traintrain_ret.mean(), traintest_ret.mean(), testtest_ret.mean()]])
    #df_ret = pd.DataFrame(ret, columns=['Train-Train', 'Train-Test', 'Test-Test'], index=['Minimal Distance', 'Mean Distance'])
    #print(df_ret)
    #epsilon_min = ret[0, :].min()/2
    #print("Epsilon: ", epsilon_min)