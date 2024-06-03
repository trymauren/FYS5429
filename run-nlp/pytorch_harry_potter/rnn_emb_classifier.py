import sys
import os
import git
path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING


class RNN_emb_classifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(RNN_emb_classifier, self).__init__()

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=False,
            nonlinearity='tanh')

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_state=None):
        if h_state is not None:
            hidden_out, h_state = self.rnn(x, h_state)
        else:
            hidden_out, h_state = self.rnn(x)
        out = self.fc(hidden_out)
        return out, h_state

    def fit(self, inputs, targets, epochs=5, lr=0.01):

        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_list = np.zeros(epochs)

        for e in tqdm(range(epochs)):

            # inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = self(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            self.loss_list[e] = loss
            loss.backward()
            optimizer.step()

        self.loss_list /= epochs
        print("Training complete.")

    def single_predict(self, seed_data, timesteps, vocab):
        device = torch.device('cpu')
        print("Using device:", device)

        self.eval()

        with torch.no_grad():
            _, _, num_features = seed_data.shape
            output, h_state = self(seed_data)

            seed_output = []
            if seed_data.shape[1] == 1:
                last_output = output[-1, 0, :]  # extract output at only time-step
            else:
                last_output = output[-2:-1, 0, :]  # extract output at last time-step

            generated_data = np.zeros((seed_data.shape[0] + timesteps, num_features))

            probabilities = nn.functional.softmax(last_output, dim=0).numpy()
            ix = np.random.choice(range(len(vocab)), p=probabilities.ravel())
            output = torch.tensor(vocab[ix], dtype=torch.float32, device=device).reshape(1, -1)

            generated_data[0] = output

            h_state = h_state[0]
            for t in range(1, timesteps + 1):
                output, h_state = self(output, h_state)
                probabilities = nn.functional.softmax(last_output, dim=0).numpy()
                ix = np.random.choice(range(len(vocab)), p=probabilities.ravel())
                output = torch.tensor(vocab[ix], dtype=torch.float32, device=device).reshape(1, -1)
                generated_data[t] = output

        return seed_output, generated_data


if __name__ == "__main__":
    device = torch.device('cpu')

    num_sequences = 500
    seq_length = 24
    word_emb = WORD_EMBEDDING()
    text_data = text_proc.read_file(path_to_root + '/data/harry_potter.txt')

    X, y = np.array(word_emb.translate_and_shift(text_data))
    X = np.array([X])

    vocab, inverse_vocab = text_proc.create_vocabulary(X)
    y = text_proc.create_labels(X, inverse_vocab)

    inputs = np.zeros((num_sequences, seq_length, X.shape[-1]))
    targets = np.zeros((num_sequences, seq_length, y.shape[-1]))
    print(inputs.shape)
    print(targets.shape)

    for i in range(num_sequences):
        input_ = X[0, i:i + seq_length, :]
        target = y[0, i + 1:i+seq_length+1, :]
        inputs[i] = input_
        targets[i] = target

    X = inputs.transpose(1, 0, 2)
    y = targets.transpose(1, 0, 2)

    print('Shape of X after batching:', X.shape)
    print('Shape of y after batching:', y.shape)

    inputs = torch.tensor(X, dtype=torch.float32, device=device)
    targets = torch.tensor(y, dtype=torch.float32, device=device)  # No squeeze needed

    # ------------ Config ------------ #
    train = False
    infer = True
    hidden_size = 400
    epochs = 1500
    learning_rate = 0.001

    # ------------ Model definition ------------ #
    model = RNN_emb_classifier(
        input_size=300,  # embedding size
        hidden_size=hidden_size,
        output_size=len(vocab)
        )
    model = model.to(device)

    # ------------ Train ------------ #
    if train:
        model.fit(inputs, targets, epochs=epochs, lr=learning_rate)
        with open(path_to_root + '/run-nlp/pytorch_harry_potter/saved_figs/loss_1', 'wb') as file:
            pickle.dump(model.loss_list, file)
        plt.figure()
        plt.plot(model.loss_list)
        plt.title('Loss over epochs')
        # plt.show()
        torch.save(model.state_dict(), path_to_root + '/run-nlp/pytorch_harry_potter/saved_models/model_1')

    # ------------ Inference ------------ #
    if infer:
        model = RNN_emb_classifier(
            input_size=300,
            hidden_size=hidden_size,
            output_size=len(vocab)
            )

        model.load_state_dict(torch.load(path_to_root + '/run-nlp/pytorch_harry_potter/saved_models/model_1'))
        model = model.to(device)

        seed_data = torch.tensor(word_emb.get_embeddings("They were")).unsqueeze(0)

        seed_output, generated = model.single_predict(seed_data, 5, vocab)
        for emb in generated:
            word = word_emb.find_closest(emb, number=1)
            print(word)

        plt.figure()
        with open(path_to_root + '/run-nlp/pytorch_harry_potter/saved_figs/loss_1', 'rb') as file:
            data = pickle.load(file)
        plt.plot(data)
        plt.title('Loss over epochs')
        plt.show()
