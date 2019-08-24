import re
import numpy as np
from os import path
from sklearn.feature_extraction.text import CountVectorizer
from random import sample

class DatasetLoader:

    @staticmethod
    def load_data(name):
        raise NotImplementedError()

    @staticmethod
    def load_data(data_type, clear_data, data_sample, n_samples, n_samples_for_class, name):
        raise NotImplementedError()

    @staticmethod
    def get_dataset_file(file_path):
        return path.join(path.dirname(__file__), *file_path)

    @staticmethod
    def sampled_data (dataset, n_samples, n_samples_for_class):
        if n_samples > len(dataset):
            n_samples = len(dataset)

        samples = []
        if n_samples_for_class == True:
            for label in list(set([item[1] for item in dataset])):
                size = len(list(filter(lambda item: item[1] == label, dataset)))
                if size < n_samples:
                    n_samples = size

            for label in list(set([item[1] for item in dataset])):
                samples = samples + sample(list(filter(lambda item: item[1] == label, dataset)), n_samples)
        else:
            samples = sample(dataset, n_samples)

        return samples

    @staticmethod
    def get_text_data(data):
        X = []
        y = []
        if type(data) is list:
            for item in data:
                if type(item) is tuple:
                    X.append(item[0])
                    y.append(item[1])
        return X, y


class Dataset:
    def __init__(self, X, y, name):
        self.X = X
        self.y = y
        self.name = name


def get_X_y (data):
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 token_pattern=r'\b\w+\b')
    docs = [x[0] for x in data]
    vectorizer.fit(docs)

    integers_from_strings = [[vectorizer.vocabulary_.get(y.lower()) for y in re.sub(r'[.!,;?]', ' ', x).split() if
                              vectorizer.vocabulary_.get(y.lower()) is not None] for x in docs]

    labels = [x[1] for x in data]
    for index, item in enumerate(integers_from_strings):
        if len(item) == 0:
            del integers_from_strings[index]
            del labels[index]

    return integers_from_strings, labels