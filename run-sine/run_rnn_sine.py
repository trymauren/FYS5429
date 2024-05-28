import sys
import git
path_to_root = git.Repo('../', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
import resource
import numpy as np
import os
#sys.path.append(os.path.abspath('..'))
from rnn.rnn import RNN
from rnn.pytorch_rnn_sine import TORCH_RNN
# from lstm.lstm import RNN
from utils.activations import Relu, Tanh
import matplotlib.pyplot as plt
from utils.loss_functions import Mean_Square_Loss
from utils.read_load_model import load_model
from datetime import datetime
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



########################################################################
########################### Sine creation ##############################
########################################################################

seq_length = 200
examples = 10


########################################################################
########################### Prediction #################################
########################################################################

seed_length = 10
time_steps_to_predict = seq_length - seed_length

epo = 5
hidden_nodes = [2,10,20,30,40,50,60]  # < 40 hidden_nodes is not able to capture periodicity

num_backsteps = seq_length

learning_rates = [0.001,0.003,0.005,0.007,0.009]
#learning_rates = [0.004]

optimisers = ['AdaGrad()', 'SGD()', 'SGD_momentum()', 'Adam()']
#optimisers = ['Adam()']
num_batches = 1

train = True
infer = True

########################################################################
########################## Script start ################################
########################################################################


np.random.seed(13)
def create_sines(examples=10, seq_length=100):
    X = []
    y = []
    for _ in range(examples):
        example_x = np.array(
            [np.random.uniform(-1, 1)*np.sin(
                np.random.uniform(-np.pi, np.pi)*np.linspace(0, 8*np.pi, seq_length+1))]
            ).T
        # example_x = np.array(
        #     [np.sin(
        #         np.linspace(0, 8*np.pi, seq_length+1))]
        #     ).T
        #example_x = np.sin(np.linspace(0, 8*np.pi, seq_length+1))
        X.append(example_x[0:-1])
        y.append(example_x[1:])

    return np.array(X), np.array(y)


# def error(validation,prediction) -> float:
#     mse = Mean_Square_Loss()
#     accuracy = mse.eval(validation, prediction)
#     return np.round(accuracy, 3)


# def model_performance_sine(rnn, num_tests = 20, test_length = 20, seed_length = 10, seq_length = 100):
#     """
#     Parameters:
#     ------------------------------------------------------
#     rnn:
#         - rnn model to measure performance on.
#     num_tests:
#         - number of predictions to base performance measure upon
#     test_length:
#         - length of each prediction
#     seed_length:
#         - length of input/seed to base a prediction upon
#     seq_length:
#         - length of the created sine waves used as data

#     Returns:
#     -------------------------------------------------------
#     Average mean square error between predicted and actual sine waves
#     across several predictions: 
#         - float
#     """
#     accuracies = []
#     X_val, y_val = create_sines(num_tests, seq_length)
#     max_start = seq_length - test_length
#     random_start = np.random.randint(0, max_start)
#     time_steps_to_predict = test_length - seed_length
#     fig, ax = plt.subplots()
#     for sine in X_val:
#         X_seed = np.array([sine[random_start:random_start + seed_length]])
#         predict = rnn.predict(X_seed,
#                               time_steps_to_generate=time_steps_to_predict).squeeze()
#         accuracies.append(error(sine[random_start
#                           + seed_length:random_start + test_length],predict))
#         ax.plot(sine[random_start + seed_length:random_start + test_length])
#         ax.plot(predict, linestyle='dotted')
#     return np.round(np.mean(accuracies), 3)

start_time = datetime.now()

X, y = create_sines(examples=examples, seq_length=seq_length)

torch_inputs = torch.tensor(X, dtype=torch.float32)
torch_targets = torch.tensor(y, dtype=torch.float32)  # No squeeze needed
torch_train_loader = DataLoader(
TensorDataset(torch_inputs, torch_targets),
batch_size=num_batches,
shuffle=False
)

X = X.reshape(1, -1, num_batches, 1)
y = y.reshape(1, -1, num_batches, 1)


X_val, y_val = create_sines(examples=1, seq_length=seq_length)

torch_val_inputs = torch.tensor(X_val, dtype=torch.float32)
torch_val_targets = torch.tensor(y_val, dtype=torch.float32)

X_val = X_val.reshape(1, -1, 1, 1)
y_val = y_val.reshape(1, -1, 1, 1)


X_seed = np.array(X_val[0][:seed_length])


for num_hidden_nodes in hidden_nodes:
    for optimiser in optimisers:
        if train:
            print(f'\n\n---------------------\nOptimiser: {optimiser.split("()")[0]}')
            print('---------------------')

        #fig_loss, axs_loss = plt.subplots(np.ceil(len(optimisers)/2), 2)
        #fig_pred, axs_pred = plt.subplots(np.ceil(len(optimisers)/2), 2)
            fig_loss = plt.figure()
            fig_loss.suptitle(f'Loss | Optimiser: {optimiser.split("()")[0]} | Hidden nodes: {num_hidden_nodes}')


        fig_pred = plt.figure()
        fig_pred.suptitle(f'Predictions | Optimiser: {optimiser.split("()")[0]} | Hidden nodes: {num_hidden_nodes}')

        # y_ticks_pos = [2]
        # y_ticks_acc = ['Error']
        
        n_rows = int(np.ceil(len(learning_rates)/2))

        for learning_rate_curr, i in zip(learning_rates, range(len(learning_rates))):
            if train:
                print(f'Training models with learning rate: {learning_rate_curr}')
                rnn = RNN(
                    hidden_activation='Tanh()',
                    output_activation='Identity()',
                    loss_function='mse()',
                    optimiser=optimiser,
                    clip_threshold=1,
                    learning_rate=learning_rate_curr,
                    name=f'./run-sine/saved_models/pretrained_rnn_{optimiser.split("()")[0]}_{learning_rate_curr}',
                    #decay_rate1 = .009,
                    #decay_rate2 = .00999
                    )
            
                hidden_state = rnn.fit(
                    X,
                    y,
                    epo,
                    num_hidden_nodes=num_hidden_nodes,
                    num_backsteps=num_backsteps,
                    num_forwardsteps=num_backsteps,
                    # X_val=X_val_batch,
                    # y_val=y_val_batch,
                )


                torch_rnn = TORCH_RNN(
                            input_size=1,
                            hidden_size=num_hidden_nodes,
                            output_size=1,
                            )

                torch_rnn.fit(torch_train_loader, epochs=epo, lr=learning_rate_curr, optimizer = optimiser)

                with open('../rnn/loss_list.pkl', 'wb') as file:
                    pickle.dump(torch_rnn.loss_list, file)
                
                ax_loss = fig_loss.add_subplot(n_rows, 2, i + 1)
                ax_loss.plot(torch_rnn.loss_list, label='Torch model')

                torch.save(torch_rnn.state_dict(), f'./saved_models/pretrained_torch_rnn_{optimiser.split("()")[0]}_{learning_rate_curr}')

                s = f'Sine_train_loss_{learning_rate_curr}.svg'
                rnn.plot_loss(plt, figax=(fig_loss, ax_loss), show=False, val=True)

                ax_loss.set_title(f'Learning rate: {learning_rate_curr}')

                fig_loss.savefig(f'./saved_figs/loss_results_{optimiser.split("()")[0]}_{learning_rate_curr}_{num_hidden_nodes}.svg')

            if infer:

                rnn = load_model(f'./saved_models/pretrained_rnn_{optimiser.split("()")[0]}_{learning_rate_curr}')
                print(X_seed.shape)
                predict,y_seed_out = rnn.predict(X_seed,
                                    time_steps_to_generate=
                                    time_steps_to_predict,
                                    return_seed_out = True
                                    )
                
                predict = predict.squeeze()
                y_seed_plot = np.array(y_seed_out).squeeze()
                plot_line = np.concatenate((y_seed_plot, predict))

                ax_pred = fig_pred.add_subplot(n_rows, 2, i + 1)
                ax_pred.plot(plot_line - 3,
                            label=f'Numpy model')
                #ax_pred.set_yticks([])
                ax_pred.set_xlabel("Time(t)")
                ax_pred.axvline(x=seed_length-1, color='black', linestyle='--')

                # y_ticks_pos.append( - offset)
                # y_ticks_acc.append(f'{np.round(error(predict, y_val[0][seed_length-1:]),3)}')

                # performance = model_performance_sine(rnn)
                # print(f'Average prediction error: {performance}\n')

                torch_rnn = TORCH_RNN(
                            input_size=1,
                            hidden_size=num_hidden_nodes,
                            output_size=1
                        )
                torch_rnn.load_state_dict(torch.load(f'./saved_models/pretrained_torch_rnn_{optimiser.split("()")[0]}_{learning_rate_curr}'))
                
                # Take the first sample of the training data for seeding
                seed_data = torch_val_inputs[0:1,0:3]
                #seed_data = torch.tensor(X_seed, dtype=torch.float32)
                # print("UU",torch_val_inputs.shape)
                seed_output, generated = torch_rnn.single_predict(seed_data, time_steps_to_predict)
                # print("AA",seed_data.shape)
                # print("BB", seed_output.shape)
                # print("CC", len(generated))
                # print(seed_data[0].squeeze().shape)
                torch_plot_line = np.concatenate((seed_data[0].squeeze(), generated))

                ax_pred.plot(torch_plot_line - 6,
                            label=f'Torch model')

                # y_ticks_pos.append( - 2*offset)
                # y_ticks_acc.append(f'{np.round(error(generated, y_val[0][seed_length-1:]),3)}')

                #print(y_val.shape)
                #y_plot_line = np.concatenate(([np.nan],y_val[0].squeeze()))
                y_plot_line = y_val[0].squeeze()
                ax_pred.plot(y_plot_line, label="True")
                ax_pred.legend()
                # ax_pred.set_yticks(y_ticks_pos, labels=y_ticks_acc)
                ax_pred.set_title(f'Learning rate : {learning_rate_curr}')

                fig_pred.savefig(f'saved_figs/pred_results_{optimiser.split("()")[0]}_{learning_rate_curr}_{num_hidden_nodes}.svg')


        # fig_loss.savefig(f'Sine_train_loss_{learning_rate_curr}.svg')
        # fig_pred.savefig(f'Sine_pred_{learning_rate_curr}.svg')
        print(f'Execution time {datetime.now() - start_time}')


bytes_usage_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
gb_usage_peak = round(bytes_usage_peak/1000000000, 3)
print('Memory consumption (peak):')
print(gb_usage_peak, 'GB')
