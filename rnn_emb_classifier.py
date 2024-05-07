import sys
import os
import git
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import utils.text_processing as text_proc
from utils.text_processing import WORD_EMBEDDING
from utils.read_load_model import load_model

path_to_root = git.Repo('.', search_parent_directories=True).working_dir
sys.path.append(path_to_root)


class RNN_emb_classifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(RNN_emb_classifier, self).__init__()

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            nonlinearity='tanh')

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_state=None):
        if h_state is not None:
            hidden_out, h_state = self.rnn(x, h_state)
        else:
            hidden_out, h_state = self.rnn(x)
        out = self.fc(hidden_out)
        return out, h_state

    def fit(self, train_loader, epochs=5, lr=0.01):
        # if torch.backends.mps.is_available():
        #     device = torch.device("mps")
        # else:
        #     device = torch.device("cpu")
        # print("Using device:", device)
        bptt_step = 2
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_list = np.zeros(epochs)

        for e in tqdm(range(epochs)):
            # for inputs, targets in train_loader:
            #     # inputs, targets = inputs.to(device), targets.to(device)
            #     outputs, _ = self(inputs)
            #     loss = criterion(outputs, targets)
            #     optimizer.zero_grad()
            #     self.loss_list[e] = loss
            #     loss.backward()
            #     optimizer.step()
            h_state = None  # Initialize hidden state to None at the start of each epoch
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                outputs, h_state = self(inputs, h_state)
                loss = criterion(outputs, targets)

                # Truncate backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Detach hidden state to prevent backpropagation through the entire dataset history
                h_state = h_state.detach()

                # Optionally reset hidden state every bptt_step batches
                if (batch_idx + 1) % bptt_step == 0:
                    h_state = None

                # Record loss for visualization or analysis later
                self.loss_list[e] += loss.item()

        # Optionally print average loss per epoch
        self.loss_list /= len(train_loader)
        print("Training complete.")

    def single_predict(self, seed_data, timesteps, vocab):
        device = torch.device('cpu')
        print("Using device:", device)

        self.eval()

        with torch.no_grad():
            _, _, num_features = seed_data.shape
            output, h_state = self(seed_data)  # seed the model (h_state)
            # seed_output = output.flatten().numpy()  # for plotting only
            seed_output = []
            if seed_data.shape[1] == 1:
                last_output = output[0, :, :]  # extract output at only time-step
            else:
                last_output = output[0, -2:-1, :]  # extract output at last time-step
            # the weird slice above is to get correct dimensions
            generated_data = np.zeros((seed_data.shape[1] + timesteps, num_features))

            # ret.append(float(last_output))
            probabilities = nn.functional.softmax(last_output, dim=1).numpy()
            ix = np.random.choice(range(len(vocab)), p=probabilities.ravel())
            output = torch.tensor(vocab[ix], dtype=torch.float32, device=device).reshape(1, -1)
            generated_data[0] = output

            for t in range(1, timesteps + 1):
                output, _ = self(output)
                probabilities = nn.functional.softmax(last_output, dim=1).numpy()
                ix = np.random.choice(range(len(vocab)), p=probabilities.ravel())
                output = torch.tensor(vocab[ix], dtype=torch.float32, device=device).reshape(1, -1)
                generated_data[t] = output

        return seed_output, generated_data


if __name__ == "__main__":
    device = torch.device('cpu')


    import tensorflow as tf
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    # Read, then decode for py2 compat.
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    # length of text is the number of characters in it

    word_emb = WORD_EMBEDDING()

    text_data = text_proc.read_file(path_to_file)

    X, y = np.array(word_emb.translate_and_shift(text_data))
    X = np.array([X])
    y = np.array([y])

    vocab, inverse_vocab = text_proc.create_vocabulary(X)
    y = text_proc.create_labels(X, inverse_vocab)

    vocab, inverse_vocab = text_proc.create_vocabulary(X)

    inputs = torch.tensor(X, dtype=torch.float32, device=device)
    targets = torch.tensor(y, dtype=torch.float32, device=device)  # No squeeze needed
    train_loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=1,
        shuffle=False
        )

    # ------------ Config ------------ #
    train = True
    infer = True
    hidden_size = 1200
    epochs = 100
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
        model.fit(train_loader, epochs=epochs, lr=learning_rate)
        with open('./rnn/loss_tf_test_pt_1.pkl', 'wb') as file:
            pickle.dump(model.loss_list, file)
        plt.figure()
        plt.plot(model.loss_list)
        plt.title('Loss over epochs')
        # plt.show()
        torch.save(model.state_dict(), './rnn/loss_tf_test_pt_1.pkl')

    # ------------ Inference ------------ #
    if infer:
        model = RNN_emb_classifier(
            input_size=300,
            hidden_size=hidden_size,
            output_size=len(vocab)
            )
        model.load_state_dict(torch.load('./rnn/loss_tf_test_pt_1.pkl'))
        model = model.to(device)
        # seed_data = inputs[0:1, 0:2, :]
        seed_data = torch.tensor(word_emb.get_embeddings("ROMEO")).unsqueeze(0)
        seed_output, generated = model.single_predict(seed_data, 5, vocab)
        for emb in generated:
            word = word_emb.find_closest(emb, number=1)
            print(word)

        # plt.figure()
        # with open('./rnn/loss_list.pkl', 'rb') as file:
        #     data = pickle.load(file)
        # plt.plot(data)
        # plt.title('Loss over epochs')
        # plt.show()
