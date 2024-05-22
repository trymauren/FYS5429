This is the repo for the our RNN project in FYS5429.
The initial goal was to build a recursive neural network (rnn) from scratch. Hopefully,
we are able to also extend the rnn implementation to a long short term memory (LSTM)
neural network implementation.

All common functions and Classes resides in `utils`. The `rnn` directory contains the rnn code,
`lstm` contains the lstm code. All data used for training the networks can be found in `Data`.


#Usage of the RNN

##Initialization
 - Text data:
    ```
    from rnn.rnn import RNN

    rnn = RNN(
            hidden_activation='Tanh()',
            output_activation='Softmax()',
            loss_function='Classification_Logloss()',
            optimiser='AdaGrad()',
            clip_threshold=np.inf,
            name='new',         <----- Optional param for storing model, more info in *Pretrained models' section
            learning_rate=0.001,
            )
    ```

 - Numeric / sine data:
    ```
    rnn = RNN(
            hidden_activation='Tanh()',
            output_activation='Identity()',
            loss_function='mse()',
            optimiser='AdaGrad()',
            clip_threshold=1,
            learning_rate=0.0001,
            )
    ```

##Training
    ###Data formatting
    - Text data:
        ```
        text_data = text_proc.read_file("data/three_little_pigs.txt")
        X, y = np.array(word_emb.translate_and_shift(text_data))
        X = np.array([X])
        y = np.array([y])
        vocab, inverse_vocab = text_proc.create_vocabulary(X)
        y = text_proc.create_labels(X, inverse_vocab)
        X = X.reshape(1, -1, 2, X.shape[-1])
        y = y.reshape(1, -1, 2, y.shape[-1])
        ```

    - Numeric / sine data:
        ```
        X, y = create_sines(examples=examples, seq_length=seq_length)

        X = X.reshape(examples, -1, num_batches, 1)
        y = y.reshape(examples, -1, num_batches, 1)
        ```

    ###Training

    - Text data:
        ```
        hidden_state = rnn.fit(
                                X,
                                y,
                                epo,
                                num_hidden_nodes=hidden_nodes,
                                num_forwardsteps=30,
                                num_backsteps=30,
                                vocab=vocab,
                                inverse_vocab=inverse_vocab,
                            )
        ```

    - Numeric / sine data:
        ```
        hidden_state = rnn.fit(
                                X,
                                y,
                                epo,
                                num_hidden_nodes=hidden_nodes,
                                num_backsteps=num_backsteps,
                                num_forwardsteps=num_backsteps,
            )
        ```
    In both cases you can add an 'X_val' and 'y_val' datasets as parameters to validate the model while training and measure validation loss, this will plot the validation loss alongside the training loss and cause early stopping should the model start performing badly on the validation data. Both these datasets should be initialized the same way as the regular training datasets, showcased in the previous section.
    The loss can be plotted by simply calling plot_loss() on the model after training:
    ```
    rnn.plot_loss(plt, show=True, val=True)
    ```

##Prediction
 - Text data:
    When predicting, the text model returns the word embedding for the most probable word for each timestep it predicts, to translate these back into words we can read, use the find_closest() method from the WORD_EMBEDDING() class imported with ```from utils.text_processing import WORD_EMBEDDING```, prediction and translation can be done as follows:
    
    ```
    predict = rnn.predict(X_seed, time_steps_to_generate=10)
    for emb in predict:
        print(word_emb.find_closest(emb, 1))
    ```
    The loop translates and prints the predicted words for each timestep.

 - Numeric / sine data:
    ```
    predict = rnn.predict(X_seed,time_steps_to_generate=10)
    ```

    The predicted data can be plotted as follows to show it:

##Pretrained models

If you have a pretrained model you can load it using the load_model() function:
```
rnn = load_model('saved_models/pretrained_model')
```
If you want to train a model and store the trained model to use it another time, you can pass a path (including the new model name) to the 'name' parameter when initializing the model before training, as follows:
```
rnn = RNN(
        hidden_activation='Tanh()',
        output_activation='Softmax()',
        loss_function='Classification_Logloss()',
        optimiser='AdaGrad()',
        clip_threshold=np.inf,
        name='/directory/of/choice/new_trained_model', <------ Path here including new model name
        learning_rate=0.001,
        )
```







