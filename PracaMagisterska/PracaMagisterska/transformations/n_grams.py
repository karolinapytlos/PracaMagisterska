from sklearn.feature_extraction.text import CountVectorizer


class NGramsTransformer():
    def __init__(self, n_grams = 2, **kwargs):
        self.vectorizer = CountVectorizer(ngram_range=(n_grams,n_grams), **kwargs)

    def fit_transform(self, raw_data):
        return self.vectorizer.fit_transform(raw_data).toarray()

    def transform(self, raw_data):
        self.vectorizer.fit(raw_data)
        return self.vectorizer.transform(raw_data).toarray()