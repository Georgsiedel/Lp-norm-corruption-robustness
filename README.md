This is the code associated with the paper 

**"Investigating the Corruption Robustness of Image Classifiers with Random Lp-norm Corruptions"** ([ArXiv](https://arxiv.org/abs/2305.05400))

run_exp.py is the test execution that calls train.py and eval.py modules from the experiments folder.

run_exp.py uses various parameters for every experiment that are defined in a config file, stores at experiments/configs (example config0 file is given and can be adjusted). Multiple config files with various numbers can be executed sequentially, they are called using the first loop in run_exp.py.

A sub-folder structure for the models is needed in experiments/models. Please notice the Readme in said folder.

The model "wrn28" used in the paper is the model defined in experiments/network.py. Any other modelname is called from torchvision.models.

experiments/samples_corrupted_img.py is used in training and testing to create Lp-norm corrupted images of arbitrary dimension, norm p and max-distance epsilon.

Visualizations from the paper (e.g. images with imperceptible Lp-norm corruptions) can be found in notebooks.
