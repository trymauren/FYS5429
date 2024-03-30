import numpy as np
import spacy


def read_file(filename : str, seq_length : int, shift : bool = False) -> np.ndarray:
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
        text_split = text.split()
        if shift:
            for i in range(1,len(text_split),seq_length):
                if i+seq_length >= len(text_split):
                    windowed_text.append(text_split[i:])
                else:
                    windowed_text.append(text_split[i:i+seq_length])
        else:
            for i in range(0,len(text_split),seq_length):
                if i+seq_length >= len(text_split):
                    windowed_text.append(text_split[i:])
                else:
                    windowed_text.append(text_split[i:i+seq_length])
    return np.array(windowed_text, dtype=object)

def read_sentence(filename : str) -> np.ndarray:
    """
    Alternative to read_file(), parses the given file into a numpy array
    containing all the sentences in the file as an entry in the array,
    in this case there are no max limits beyond pythons capabilities to
    the length of the passed file as each sentence in the file will be
    translated to tokenized embedding vectors separately.

    Parameters:
    --------------------------------
    filename : str
    - The name of the text file to be parsed as a string

    Returns:
    --------------------------------
    numpy.ndarray
    - All sentences in the passed text file as a separate string entry 
      in the array. The text is split into sentences in the whitespace 
      following any "!", "?" or "."
    """
    text = ""
    with open(filename,"r") as f:
        text = f.read()
        #TODO: this split is defo not correct lol, have to change to 
        #split in the whitespace right after !,? or .
        text = text.split(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<=\.|\?)")
    return np.array(text)



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
    
if __name__ == "__main__":
    res = read_file("embedding_test.txt", 5)
    print(res)
    emb_obj = WORD_EMBEDDING()
    emb = emb_obj.get_embeddings(str(res))
    for token in emb:
        print(emb_obj.find_closest(token, 1))
        print(token)
