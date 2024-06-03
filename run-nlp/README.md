# This folder is the home of the natural language models and programs.

The xx\_mp.py files (in the directories in this directory) can be run to replicate experiments conducted.
They use the multiprocessing library to run models with different configurations in parallel.
Running them on a standard laptop can therefore take some time.

The xx.py files can be used for inference after models are trained. Some trained
models have been pushed to github, others are only reciding locally (due to them being
too large for github). Have a look in the `saved_models` folder (under `harry_potter` or other)
to see which models are already trained. Hyperparameters must be changed in the file.

All .py files can be ran from anywhere in the repo.