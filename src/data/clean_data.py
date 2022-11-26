from typing import List

# russian tokenizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
# russian stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('russian')
# russian lemmatizer
# from pymystem3 import Mystem
# mystem = Mystem()
# lemmatizer = lambda x: mystem.lemmatize(x)[0]
# russian stemmer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")
import pandas as pd


class TextCleaning:
    """
    Applies tokenizer and remove emojis from data
    Optionally you can use lemmatizer and stopwords
    """
    def __init__(self, tokenizer, stemmer=None, lemmatizer=None, stop_words=None):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.lemmatizer = lemmatizer
        self.stop_words = stop_words

    def preprocessing(self, data: List[str]) -> List[List[str]]:
        """
        Do all preprocessing
        :param data: List[str]
        :return: clean_data: List[List[str]]
        """
        clean_data = [self.tokenizer.tokenize(string.lower()) for string in data]
        if self.stop_words:
            for string in clean_data:
                clean_data[clean_data.index(string)] = [word for word in string if word not in self.stop_words]
        if self.lemmatizer:
            for string in clean_data:
                clean_data[clean_data.index(string)] = [self.lemmatizer.lemmatize(word) for word in string]
        if self.stemmer:
            for string in clean_data:
                clean_data[clean_data.index(string)] = [self.stemmer.stem(word) for word in string]
        return clean_data

if __name__ == "__main__":
    labeled = pd.read_csv('../../data/interim/labeled.csv')
    unlabeled = pd.read_csv('../../data/interim/unlabeled.csv')

    Cleaner = TextCleaning(tokenizer, stemmer, stop_words=stop_words)
    for df in [labeled, unlabeled]:
        print("cleaning data")
        for col in ['Position', 'content', 'description', 'profession', 'profession_desc']:
            print(f"cleaning {col}")
            cleaned = Cleaner.preprocessing(df[col].astype(str))
            df[col] = [" ".join(data_) for data_ in cleaned]

    print("saving data")
    labeled.to_csv('../../data/interim/labeled.csv', index=False)
    unlabeled.to_csv('../../data/interim/unlabeled.csv', index=False)

