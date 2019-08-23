from sklearn.feature_extraction.text import CountVectorizer


class BagOfWordsTransformer():
    def __init__(self, **kwargs):
        self.vectorizer = CountVectorizer(**kwargs)

    def fit_transform(self, raw_data):
        test = self.vectorizer.fit_transform(raw_data).toarray()

        print("len(vectorizer.get_feature_names()): ", len(self.vectorizer.get_feature_names()))
        print("len(vectorizer.vocabulary_): ", len(self.vectorizer.vocabulary_))

        return test

    def transform(self, raw_data):
        self.vectorizer.fit(raw_data)
        return self.vectorizer.transform(raw_data).toarray()