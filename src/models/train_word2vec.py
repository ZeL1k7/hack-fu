from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import gensim

from gensim.models import KeyedVectors


class WordVec:
    def __init__(self):
        self.w2v_model = Word2Vec(
            min_count=10,
            window=5,
            vector_size=300,
            negative=10,
            alpha=0.03,
            min_alpha=0.0007,
            sample=6e-5,
            sg=1,
            workers=4)

    def train(self, vocab_data, train_data, save_model: bool = True):
        w2v_model = self.w2v_model
        w2v_model.build_vocab(vocab_data)
        w2v_model.train(train_data, total_examples=w2v_model.corpus_count, epochs=1, report_delay=1)
        if save_model:
            w2v_model.save("word2vec.model")


if __name__ == "__main__":
    data_labeled = pd.read_csv('../../data/interim/labeled.csv')
    data_unlabeled = pd.read_csv('../../data/interim/unlabeled.csv')
    print("read data")
    data_labeled = data_labeled.drop(['index', 'target'], axis=1)
    data_unlabeled = data_unlabeled.drop(['index', 'target', 'profession', 'profession_desc'], axis=1)
    print("droped columns")
    data = np.append(data_unlabeled, data_labeled)
    data = [str(data_) for data_ in data]
    data = [data_.split(' ') for data_ in data]
    print("data loaded")
    w2v = WordVec()
    print('w2v inited')
    w2v.train(data, data)
    print('w2v trained')
