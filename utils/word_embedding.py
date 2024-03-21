import os
import subprocess
import numpy as np
import chromadb
import spacy


class word_embedding():
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_lg")
        import en_core_web_lg
        self.nlp = en_core_web_lg.load()
    
    def read_txt(self,file) -> np.array:
        with open(file, 'r') as f:
            paragraphs = f.split("\n\n")
            paragraph_embeddings = []
            for pg in paragraphs:
                paragraph_embeddings.append(self.get_embeddings(pg))
        return np.array(paragraph_embeddings)


    def get_embeddings(self, text: str) -> np.ndarray:
        embeddings = []
        doc = self.nlp(text)
        for token in doc:
            embeddings.append(token.vector)
        return embeddings

    def retrieve_word(self, embedding_obj: np.ndarray) -> str:
        return embedding_obj.text
    
    def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) \
        -> float:
        return embedding1.similarity(embedding2)
    
    def find_closest(self, embedding: np.ndarray, number: int) -> np.ndarray:
        most_similar = self.nlp.vocab.vectors.most_similar(
                                                        np.ndarray([embedding]), 
                                                        n=number)
        keys = most_similar[0][0]
        nearest_words = []
        for key in keys: 
            nearest_words.append(self.nlp.vocab.strings[key])
        return nearest_words
    

    
if __name__ == "__main__":
    w_emb = word_embedding()
    dog = w_emb.get_embeddings("dog log mog smog fog clog")
    cat = w_emb.get_embeddings("cat")
    apple = w_emb.get_embeddings("apple")
    print(dog)
