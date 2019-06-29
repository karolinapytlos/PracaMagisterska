from SimilarityForest import SimilarityForest
from sklearn.model_selection import train_test_split
import Functions as fn
import Transformations as ts
from datasets.airline_tweets import AirlineTweetsDataset

dataset = AirlineTweetsDataset.load_data()

tfidf = ts.BagOfWords()
tfidf_vectors = tfidf.convert_to_tfidf_vectors(dataset.X)

X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, dataset.y, test_size=0.20)

sf = SimilarityForest(2, fn.cosie_similarity, 2)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)