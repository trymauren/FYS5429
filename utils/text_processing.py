import numpy as np
import spacy


def read_file(filename : str) -> np.ndarray:
    text = ""
    with open(filename, 'r') as f:
        text = f.read()
        len_chars = sum(len(word) for word in text.strip().split())
        print(len_chars)
        if len_chars > 100000:
            raise ValueError("Text file can't contain more than 100 000\
                             characters!")
    # solves problem with empty strings left in res.
    #   -> probably not very scalable. consider handling differently
    return np.array(text)

class word_embedding():
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_lg")
        import en_core_web_lg
        self.nlp = en_core_web_lg.load()
    
    #def read_txt(self,file) -> np.array:
    #    with open(file, 'r') as f:
    #        paragraphs = f.split("\n\n")
    #        paragraph_embeddings = []
    #        for pg in paragraphs:
    #            paragraph_embeddings.append(self.get_embeddings(pg))
    #    return np.array(paragraph_embeddings)


    def get_embeddings(self, text: str) -> np.ndarray:
        embeddings = []
        doc = self.nlp(text)
        for token in doc:
            embeddings.append(token.vector)
        return embeddings

    #def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) \
    #    -> float:
    #    return embedding1.similarity(embedding2)
    
    def find_closest(self, embedding: np.ndarray, number: int) -> np.ndarray:
        most_similar = self.nlp.vocab.vectors.most_similar(
                                                        np.array([embedding]), 
                                                        n=number)
        keys = most_similar[0][0]
        nearest_words = []
        for key in keys: 
            nearest_words.append(self.nlp.vocab.strings[key])
        return nearest_words
    
if __name__ == "__main__":
    res = read_file("embedding_test.txt")
    # print(res, type(res))
    emb_obj = word_embedding()
    emb = emb_obj.get_embeddings(str(res))
    for token in emb:
        print(emb_obj.find_closest(token, 1))
        print(token)
