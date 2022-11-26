from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import euclidean_distances
import pickle

def make_embeddings2(okz):
    global embeddings2 
    i = 0
    for okz_ in okz['description']:
        print(okz_)
        embeddings2.append(model.encode([okz_]))
        i += 1
        print(i)

    with open('embeddings2.pickle', 'wb') as f:
        pickle.dump(embeddings2, f)


def load_embeddings2():
    global embeddings2
    with open('embeddings2.pickle', 'rb') as f:
        embeddings2 = pickle.load(f)


def make_dict_score(position, okz):
    global j 
    j += 1
    print(f'{j}/33260')
    d_iou = {}
    embeddings1 = model.encode([position])
    for i in range(len(okz)):
        score = cosine_similarity(embeddings1, embeddings2[i])
        # score = euclidean_distances(embeddings1, embeddings2[i])
        if score > 1e-8:
            d_iou[okz.code[i]] = score[0][0]
    print(d_iou)
    return d_iou
        

if __name__ == '__main__':
    okz = pd.read_csv('../../data/interim/okz.csv', sep=',')
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    embeddings2 = []
    j = 0

    load_embeddings2()
    print('done preparing')

    unlabeled = pd.read_csv('../../data/interim/unlabeled.csv')
    okz = pd.read_csv('../../data/interim/okz.csv', sep=',')

    # unlabeled['target'] = unlabeled['Position'].apply(lambda x: make_dict_score(x, okz))
    # print(unlabeled.head())
    # unlabeled.to_csv('../../data/interim/unlabeled_fixed.csv', index=False)
    # position = unlabeled['Position'][2]

    position = unlabeled['Position'][0] + ' ' +  unlabeled['content'][0]
    print(position)
    score_dict = make_dict_score(position, okz)
    print(score_dict)

    # find max score
    print("FINDING MAX SCORE")
    max_score = 0
    max_code = ''
    for code, score in score_dict.items():
        if score > max_score:
            max_score = score
            max_code = code
    print(max_code)
    print(okz[okz['code'] == max_code]['description'].values[0])