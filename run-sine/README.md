
# Reproduction of Sine Predictions
This README covers how to run our pretrained RNN model for sine predictions to reproduce and verfiy some of the results we have showcased in our paper. For this purpose we have a premade python script which retrieves the trained model and runs the predictions for you.

## Usage
Just run the run_pretrained_sine.py script from the CLI using the command ´´´python3 run_pretrained_sine.py´´´ when standing in the ´´´/FYS5429/run-sine´´´ directory, the script is has default parameters set to retrieving all the Numpy and PyTorch RNN models with 10 or 50 hidden nodes trained with Adam at any tested learning rates used in all 3 tests in the paper.

The script goes through each test, 'random, 'noisy' and 'plain' in that order. First it plots and visualizes the dataset the models were trained on, then shows the predictions for the models. The predictions are shown in the same way as in the paper the Numpy model again the PyTorch model with a validation wave on top, for all the different learning rates. One plot for 10 hidden nodes, and the other for 50 hidden nodes, the process then repeats, and you can compare the plots to the ones in the paper to validate their integrity.

It is possible to recreate the results from all tests done in the paper, however, to limit the amount of storage required to pull the git repo we chose to only store these few pretrained models that's defaulted in the ´´´run_pretrained_sine.py´´´ script, and you'll have to train the models for all the other configurations if you want to validate any of the other results showcased in the paper. See the README.md file in the FYS5429 directory for instructions on how to train and use a Numpy RNN model. 

NB: It should be noted that due to some variantions in versioning and platforms, the randomness in data and initializations may wary a bit if you try to recreate our results, for both the Numpy-, and the PyTorch model.