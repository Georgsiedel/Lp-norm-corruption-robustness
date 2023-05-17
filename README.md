This is the code associated with the paper 

**"Investigating the Corruption Robustness of Image Classifiers with Random Lp-norm Corruptions"**

published as a preprint version on [Researchgate](https://www.researchgate.net/publication/370617296_Investigating_the_Corruption_Robustness_of_Image_Classifiers_with_Random_Lp-norm_Corruptions) and [arXiv](https://arxiv.org/abs/2305.05400).


run_exp.py is the test execution that calls train.py and eval.py modules from the experiments folder.

run_exp.py uses various parameters for every experiment that are defined in a config file, stores at experiments/configs (example config0 file is given and can be adjusted).

multiple config files with various numbers can be executed sequentially, they are called using the first loop in run_exp.py.

The model "wrn28" used in the paper is the model defined in experiments/network.py. Any other modelname is called from torchvision.models.

experiments/samples_corrupted_img.py is used in training and testing to create Lp-norm corrupted images of arbitrary dimension, norm p and max-distance epsilon.

Visualizations from the paper (e.g. images with imperceptible Lp-norm corruptions) can be found in notebooks.
