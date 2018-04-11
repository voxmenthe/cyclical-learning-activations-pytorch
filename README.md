# Pytorch Implementation of multiple Resnet-Style Architectures
## Allows comparison training with several custom activations
## Allows testing with the AddSign and PowerSign optimizers
## Various learning rate schedules including Cyclical Learning Rate (within epoch) are implemented
## Fairly comprehensive logging of training statistics
## Automated testing of multiple models and multiple activations/hyperparameters

## How To Run:
<code>python3 main_multi.py --epochs 10</code>
* Some more examples can be seen in the `TRAINING_INSTRUCTIONS.md` file.
* While the project is set up to use command-line arguments, many of these are overridden in the `hyperparams.py` file, so strongly suggest using that for comprehensive experiments.

## Models and Model Configuration
* All of the main model architectures are stored in the **models** directory, each in their own separate file.
* Each model file must be registered in <code>models/\_\_init\_\_.py</code>
* The actual model names are at the bottom of each model file, and must be typed in to the `hyperparams.py` file.

## Activation Functions
* Several (but not all) of the models support using custom activation functions. The ones that do are mostly denoted with an `_act` suffix. 
* The file custom_activations.py includes my implementations of a few new activation functions from recent research that are not yet implemented in PyTorch. There are also a few experiments of my own, some of them promising, to be written up in a blog post soon.

## Training Schedules
* There are several different training schemes implemented. I expect the most common ones to be:
1. '--lr-fixed-stages' which simply divides the total number of epochs to train into a number of 'stages' and divides '--lr' by '--lr-increment' at the end of each stage.
2. '--inepoch-lr-decay' - this is roughly equivalent to the "Cyclical Learning Rate" described in [Leslie, 2018 - https://arxiv.org/abs/1803.09820] where you set a max and min learning rate, as well as optional decay factors for each, and then the learning rate starts high and decays linearly within each epoch. This approach tends to work well with SGD, AddSign and PowerSign optimizers.

## Logging and Checkpoints
* A csv file of training data for each model trained is saved to the <code>'/results/train'</code> and <code>'/results/test'</code> folders (or folders you specify).
* If multiple models are trained, combined data is also saved in the <code>'/results/combo_train'</code> and <code>'/results/combo_test'</code> folders while summary data is saved in the <code>'/results/train_summary'</code> and <code>'/results/test_summary'</code> folders.
* This project also includes a standard progress bar in the terminal window which has detailed information, but defaults to the basics. If your screen is wide enough, you can try to set --long-progress-bar to see more detailed information.

## Inference
I have included some basic inference code in the **inference.ipynb** Jupyter notebook