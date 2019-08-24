from sklearn.feature_extraction.text import CountVectorizer


class BagOfWordsTransformer():
    def __init__(self, **kwargs):
        self.vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', **kwargs)

    def fit_transform(self, raw_data):
        return self.vectorizer.fit_transform(raw_data).toarray()

    def transform(self, raw_data):
        self.vectorizer.fit(raw_data)
        return self.vectorizer.transform(raw_data).toarray()