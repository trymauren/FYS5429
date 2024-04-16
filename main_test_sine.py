import sys
import git
import numpy as np
import jax.numpy as jnp
from jax import random
from rnn.rnn import RNN
from utils.activations import Relu, Tanh
import matplotlib.pyplot as plt
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


def create_sines(examples=10, seq_length=100):
    X = []
    y = []
    key = random.key(0)             # 0??
    for _ in range(examples):
        key, subkey = random.split(key)
        example_x = jnp.array(
            [random.uniform(subkey, minval=-2, maxval=2)*jnp.sin(
                jnp.linspace(0, 4*jnp.pi, seq_length+1))]
            ).T
        X.append(example_x[0:-1])
        y.append(example_x[1:])

    return jnp.array(X), jnp.array(y)


# Sine creation
seq_length = 20
examples = 100

# Prediction
seed_length = 2
time_steps_to_predict = seq_length - seed_length

# RNN init
epo = 10
hidden_nodes = 100
num_backsteps = seq_length
# learning_rates = [0.001,0.003,0.005,0.007,0.009]
learning_rates = [0.003]
# optimisers = ['AdaGrad()', 'SGD()', 'SGD_momentum()','RMSProp()']
optimisers = ['AdaGrad()']

# Plotting
offset = 3

X, y = create_sines(examples=examples, seq_length=seq_length)

# Plotting the sine waves that are passed as training data
plt.title("Randomized sines used for training")
plt.ylabel("Amplitude(y)")
plt.xlabel("Time(t)")

for sine in X:
    plt.plot(sine)
plt.savefig(f'Sine_training_data | size = {examples}')

X_val, y_val = create_sines(examples=1, seq_length=seq_length)
X_seed = jnp.array([X_val[0][:seed_length]])


for learning_rate_curr in learning_rates:
    print(f'\n\n---------------------\nlearning rate: {learning_rate_curr}')
    print('---------------------')

    fig_loss, ax_loss = plt.subplots()
    fig_pred, ax_pred = plt.subplots()

    ax_pred.set_title(f"Predictions | learning rate: {learning_rate_curr}")
    ax_pred.set_yticks([])
    ax_pred.set_xlabel("Time(t)")
    ax_pred.axvline(x=seed_length-1, color='black', linestyle='--')

    for optimiser, i in zip(optimisers, range(len(optimisers))):
        rnn = RNN(
            hidden_activation='Tanh()',
            output_activation='Identity()',
            loss_function='mse()',
            optimiser=optimiser,
            clip_threshold=1,
            learning_rate=learning_rate_curr,
            )

        hidden_state = rnn.fit(
            X, y, epo,
            num_hidden_nodes=hidden_nodes, return_sequences=True,
            num_backsteps=num_backsteps)

        rnn.plot_loss(plt, figax=(fig_loss, ax_loss), show=False)

        predict = rnn.predict(X_seed,
                              time_steps_to_generate=time_steps_to_predict)

        plot_line = jnp.concatenate((X_seed[0], predict))
        ax_pred.plot(plot_line - (i+1)*offset,
                     label=str(rnn._optimiser.__class__.__name__))
        ax_pred.legend()

    ax_pred.plot(X_val[0], label="X val")
    ax_pred.legend()

    ax_loss.set_title(f"Training loss | learning rate: {learning_rate_curr}")

    fig_loss.savefig(f'Sine_train_loss_{learning_rate_curr}.svg')
    plt.savefig(f'Sine_pred_{learning_rate_curr}.svg')
