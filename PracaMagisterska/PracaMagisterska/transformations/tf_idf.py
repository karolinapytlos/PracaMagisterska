from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfTransformer():
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(use_idf=True, **kwargs)

    def fit_transform(self, raw_data):
        return self.vectorizer.fit_transform(raw_data).toarray()

    def transform(self, raw_data):
        self.vectorizer.fit(raw_data)
        return self.vectorizer.transform(raw_data).toarray()