import numpy as np
import re
from word_embedding import word_embedding

def read_file(filename : str) -> np.ndarray:
    res = list
    with open(filename, 'r') as txt_file:
        # res is a list of sentences without punctutation at the end.
        #   - sentences are of type numpy.str_
        res = [
            s.strip() for s in re.split(r"\.|\?|\!|\n", txt_file.read())
        ]
    # solves problem with empty strings left in res.
    #   -> probably not very scalable. consider handling differently
    while "" in res:
        res.remove("")
    return np.asarray(res)

if __name__ == "__main__":
    res = read_file("test.txt")
    # print(res, type(res))
    emb_obj = word_embedding()
    for sentence in res:
        emb = emb_obj.get_embeddings(str(sentence))
        # print(emb_obj.find_closest(emb, 10))
        for embedding in emb:
            print(emb_obj.retrieve_word(embedding))
    