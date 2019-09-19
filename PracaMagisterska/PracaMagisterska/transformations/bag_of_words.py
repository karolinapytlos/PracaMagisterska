from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

class BagOfWordsTransformer():
    def __init__(self, normalize, **kwargs):
        self.vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', **kwargs)

    def fit_transform(self, raw_data):
		if self.normalize == True:
			return normalize(self.vectorizer.fit_transform(raw_data).toarray(), norm='max', axis=0)
        return self.vectorizer.fit_transform(raw_data).toarray()

    def transform(self, raw_data):
        self.vectorizer.fit(raw_data)
		if self.normalize == True:
			return normalize(self.vectorizer.transform(raw_data).toarray(), norm='max', axis=0)
        return self.vectorizer.transform(raw_data).toarray()