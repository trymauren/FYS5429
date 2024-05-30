import numpy as np
import spacy


def read_file(filename : str) -> np.ndarray:
    """
    Parses the given file into a numpy array containing one long string
    up to 100 000 charcters long, the max length is set to be within
    Spacy's max tokenization length.
    Parameters:
    --------------------------------
    filename : str
    - The name of the text file to be parsed as a string

    Returns:
    --------------------------------
    numpy.ndarray
    - the text as a single long string as an entry in the array
    """
    text = ""
    windowed_text = []
    with open(filename, 'r') as f:
        text = f.read()
        text = text.replace("\n", "")
        len_chars = sum(len(word) for word in text.strip().split())
        if len_chars > 100000:
            text = text[:50000]
            # raise ValueError("Text file can't contain more than 100 000\
            #                  characters!")
    return text

def create_vocabulary(word_embeddings : np.ndarray) -> dict:
    unique_embeddings = np.unique(word_embeddings[0], axis=0)
    vocabulary = dict(zip(range(len(unique_embeddings)), unique_embeddings))
    inverse_vocabulary = dict(zip(tuple(map(tuple,unique_embeddings)),
                                  range(len(unique_embeddings))))
    return vocabulary, inverse_vocabulary

def create_labels(X,inverse_vocabulary) -> np.ndarray:
    y = []
    for embedding in X[0]:
        y.append([inverse_vocabulary[tuple(embedding)]])
    y = np.array(y)
    one_hot_y = np.zeros((len(y),y.max()+1))
    for value, i in zip(y, range(len(y))):
        one_hot_y[i,value] = 1
    return np.array([one_hot_y])


class WORD_EMBEDDING():
    """
    Class for initializing a word embedding dataset and processing given text
    to and from word embeddings.
    """
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_lg")
        import en_core_web_lg
        self.nlp = en_core_web_lg.load()

    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Translates the given string into word embeddings/vector by tokenizing
        the string and then translating the tokens into word embeddings

        Parameters:
        --------------------------
        text : str
        - as string of the text/word that's to be tokenized and translated into
          word embeddings

        Returns:
        --------------------------
        np.ndarray:
        - numpy array of word embeddings/1d vectors representing
          each token as a numerical vector
        """
        embeddings = []
        doc = self.nlp(text)
        for token in doc:
            embeddings.append(token.vector)
        return np.array(embeddings)

    #def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) \
    #    -> float:
    #    return embedding1.similarity(embedding2)

    def find_closest(self, embedding: np.ndarray, number: int) -> np.ndarray:
        """
        Finds the words/tokens in the word embedding dataset that's 
        closest to the passed word embedding in the vector space 
        using euclidean distance

        Parameters:
        ---------------------------
        embedding : np.ndarray
        - A 1d word embedding vector of length 300
        
        number : int
        - parameter specifiying how many of the n closest words to given word 
          embedding vector to show
        
        Returns:
        ---------------------------
        np.ndarray:
        - numpy array containing the n closest words the the given word
          embedding
        """
        most_similar = self.nlp.vocab.vectors.most_similar(
                                                        np.array([embedding]), 
                                                        n=number)
        keys = most_similar[0][0]
        nearest_words = []
        for key in keys: 
            nearest_words.append(self.nlp.vocab.strings[key])
        return np.array(nearest_words)
    
    def translate_and_shift(self, data: str):
        """
        Translate text data from string into 2 embedding data sets, the
        first ranging from 0->(N-1), and the second acting as a validation set,
        ranging from 1 -> N
        
        Parameters:
        ---------------------------
            data : str
            - text data as a string up to 100 000 characters in length

        Returns:
        ---------------------------
            x : np.ndarray
            - text data translated into embeddings, covers 0->(len(X)-1) of X

            y : np.ndarray
            - text data translated into embeddings, covers 1->len(X) of X, acts
              as validation set for x
        """
        word_embs = self.get_embeddings(data)
        x = word_embs[0:-1]
        y = word_embs[1:]
        return x,y


if __name__ == "__main__":
    res = read_file("embedding_test.txt", 5)
    print(res)
    emb_obj = WORD_EMBEDDING()
    emb = emb_obj.get_embeddings(str(res))
    for token in emb:
        print(emb_obj.find_closest(token, 1))
        print(token)
