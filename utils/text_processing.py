import numpy as np
import jax.numpy as jnp
import spacy


def read_file(filename : str) -> jnp.ndarray:
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
        len_chars = sum(len(word) for word in text.strip().split())
        if len_chars > 100000:
            raise ValueError("Text file can't contain more than 100 000\
                             characters!")
    return text



class WORD_EMBEDDING():
    """
    Class for initializing a word embedding dataset and processing given text
    to and from word embeddings.
    """
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_lg")
        import en_core_web_lg
        self.nlp = en_core_web_lg.load()

    def get_embeddings(self, text: str) -> jnp.ndarray:
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
        jnp.ndarray:
        - numpy array of word embeddings/1d vectors representing
          each token as a numerical vector
        """
        embeddings = []
        doc = self.nlp(text)
        for token in doc:
            embeddings.append(token.vector)
        return jnp.array(embeddings)

    #def get_similarity(self, embedding1: jnp.ndarray, embedding2: jnp.ndarray) \
    #    -> float:
    #    return embedding1.similarity(embedding2)
    
    def find_closest(self, embedding: jnp.ndarray, number: int) -> jnp.ndarray:
        """
        Finds the words/tokens in the word embedding dataset that's 
        closest to the passed word embedding in the vector space 
        using euclidean distance

        Parameters:
        ---------------------------
        embedding : jnp.ndarray
        - A 1d word embedding vector of length 300
        
        number : int
        - parameter specifiying how many of the n closest words to given word 
          embedding vector to show
        
        Returns:
        ---------------------------
        jnp.ndarray:
        - numpy array containing the n closest words the the given word
          embedding
        """
        most_similar = self.nlp.vocab.vectors.most_similar(
                                                        jnp.array([embedding]), 
                                                        n=number)
        keys = most_similar[0][0]
        nearest_words = []
        for key in keys: 
            nearest_words.append(self.nlp.vocab.strings[key])
        return jnp.array(nearest_words)
    
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
            x : jnp.ndarray
            - text data translated into embeddings, covers 0->(len(X)-1) of X

            y : jnp.ndarray
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
