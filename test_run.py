from rnn_not_nonsense import rnn_not_nonsense
from utils.word_embedding import *
import numpy as np

rnn = rnn_not_nonsense(train=False)
word_embeddings = word_embedding()
#X_data = word_embedding.get_embeddings(word_embeddings,
#"There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.\
#There was a big black cat in the house on top of the small green hill, it does what it will.")

X_data = np.array([np.sin(np.linspace(0,4*np.pi,100))]).T
print(X_data)
rnn.fit(X=X_data,
        y = X_data,
        epochs = 1000,
        num_hidden_states = 5,
        num_hidden_nodes = 5
        )
#pred_x = word_embedding.get_embeddings(word_embeddings,"There was a big black cat in the house")
pred_x = np.array([np.sin(np.linspace(0,4*np.pi,100))]).T
print(rnn.predict(pred_x))