from typing import List


class TextCleaning:
    """
    Applies tokenizer and remove emojis from data
    Optionally you can use lemmatizer and stopwords
    """
    def __init__(self, data: List[str], tokenizer, stemmer=None, lemmatizer=None, stop_words=None):
        self.data = data
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
