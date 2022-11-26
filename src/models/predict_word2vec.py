import gensim
import re

import numpy as np

from src import TextCleaning
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import pandas as pd

if __name__ == "__main__":

    html_regexp = re.compile(r'(\<(/?[^>]+)>)')

    df_train = pd.read_csv('../../data/raw/train.csv')
    df_train = df_train[df_train['target'] != -1]
    df_train['description'] = df_train['description'].astype(str).apply(lambda x: html_regexp.sub(r'', x))

    tokenizer = WordPunctTokenizer()
    stemmer = SnowballStemmer("russian")
    stop_words = stopwords.words('russian')
    cleaner = TextCleaning(tokenizer, stemmer, stop_words=stop_words)

    clean_name = cleaner.preprocessing(df_train['name'].values)
    clean_description = cleaner.preprocessing(df_train['description'].values)

    word2vec = gensim.models.Word2Vec.load("word2vec.model")

    vectors_name = np.array([])
    for string in clean_name[:2]:
        for word in string:
            vectors_name = np.append(vectors_name, word2vec.wv.get_vector(word))

    y = df_train['target'].values






