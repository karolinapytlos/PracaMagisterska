from SimilarityForest import SimilarityForest
from sklearn.model_selection import train_test_split
import Functions as fn
import Transformations as ts

from datasets.airline_tweets import AirlineTweetsDataset
from datasets.blog_authorship_corpus import BlogAuthorshipCorpusDataset
from datasets.clothing_fit_data import ClothingFitDataset
from datasets.jeopardy_questions import JeopardyQuestionsDataset
from datasets.movie_reviews import MovieReviewsDataset
from datasets.sms_spam_collection import SmsSpamCollectionDataset
from datasets.social_recommendation_data import SocialRecommendationDataset

n_trees = 5
n_pairs = 3
bag = ts.BagOfWords()
test_size = 0.20

# TEST AirlineTweetsDataset
print(" ---- AirlineTweetsDataset ---- ")
dataset = AirlineTweetsDataset.load_data("sample.txt")
print("dataset.X: ", len(dataset.X))
print("dataset.y: ", len(dataset.y))

print(" ---- bag of words ---- ")
bag_vectors = bag.convert_to_bow_vectors(dataset.X)
print("bag_vectors: ", len(bag_vectors))

X_train, X_test, y_train, y_test = train_test_split(bag_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)

print(" ---- tf-idf ---- ")
tfidf_vectors = bag.convert_to_tfidf_vectors(dataset.X)
print("tfidf_vectors: ", len(tfidf_vectors))

X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)


# TEST BlogAuthorshipCorpusDataset
print(" ---- BlogAuthorshipCorpusDataset ---- ")
dataset = BlogAuthorshipCorpusDataset.load_data("sample.txt")
print("dataset.X: ", len(dataset.X))
print("dataset.y: ", len(dataset.y))

print(" ---- bag of words ---- ")
bag_vectors = bag.convert_to_bow_vectors(dataset.X)
print("bag_vectors: ", len(bag_vectors))

X_train, X_test, y_train, y_test = train_test_split(bag_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)

print(" ---- tf-idf ---- ")
tfidf_vectors = bag.convert_to_tfidf_vectors(dataset.X)
print("tfidf_vectors: ", len(tfidf_vectors))

X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)


# TEST ClothingFitDataset
print(" ---- ClothingFitDataset ---- ")
dataset = ClothingFitDataset.load_data("sample.txt")
print("dataset.X: ", len(dataset.X))
print("dataset.y: ", len(dataset.y))

print(" ---- bag of words ---- ")
bag_vectors = bag.convert_to_bow_vectors(dataset.X)
print("bag_vectors: ", len(bag_vectors))

X_train, X_test, y_train, y_test = train_test_split(bag_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)

print(" ---- tf-idf ---- ")
tfidf_vectors = bag.convert_to_tfidf_vectors(dataset.X)
print("tfidf_vectors: ", len(tfidf_vectors))

X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)


# TEST JeopardyQuestionsDataset
print(" ---- JeopardyQuestionsDataset ---- ")
dataset = JeopardyQuestionsDataset.load_data("sample.txt")
print("dataset.X: ", len(dataset.X))
print("dataset.y: ", len(dataset.y))

print(" ---- bag of words ---- ")
bag_vectors = bag.convert_to_bow_vectors(dataset.X)
print("bag_vectors: ", len(bag_vectors))

X_train, X_test, y_train, y_test = train_test_split(bag_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)

print(" ---- tf-idf ---- ")
tfidf_vectors = bag.convert_to_tfidf_vectors(dataset.X)
print("tfidf_vectors: ", len(tfidf_vectors))

X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)


# TEST MovieReviewsDataset
print(" ---- MovieReviewsDataset ---- ")
dataset = MovieReviewsDataset.load_data("sample.txt")
print("dataset.X: ", len(dataset.X))
print("dataset.y: ", len(dataset.y))

print(" ---- bag of words ---- ")
bag_vectors = bag.convert_to_bow_vectors(dataset.X)
print("bag_vectors: ", len(bag_vectors))

X_train, X_test, y_train, y_test = train_test_split(bag_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)

print(" ---- tf-idf ---- ")
tfidf_vectors = bag.convert_to_tfidf_vectors(dataset.X)
print("tfidf_vectors: ", len(tfidf_vectors))

X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)


# TEST SmsSpamCollectionDataset
print(" ---- SmsSpamCollectionDataset ---- ")
dataset = SmsSpamCollectionDataset.load_data("sample.txt")
print("dataset.X: ", len(dataset.X))
print("dataset.y: ", len(dataset.y))

print(" ---- bag of words ---- ")
bag_vectors = bag.convert_to_bow_vectors(dataset.X)
print("bag_vectors: ", len(bag_vectors))

X_train, X_test, y_train, y_test = train_test_split(bag_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)

print(" ---- tf-idf ---- ")
tfidf_vectors = bag.convert_to_tfidf_vectors(dataset.X)
print("tfidf_vectors: ", len(tfidf_vectors))

X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)


# TEST SocialRecommendationDataset
print(" ---- SocialRecommendationDataset ---- ")
dataset = SocialRecommendationDataset.load_data("sample.txt")
print("dataset.X: ", len(dataset.X))
print("dataset.y: ", len(dataset.y))

print(" ---- bag of words ---- ")
bag_vectors = bag.convert_to_bow_vectors(dataset.X)
print("bag_vectors: ", len(bag_vectors))

X_train, X_test, y_train, y_test = train_test_split(bag_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)

print(" ---- tf-idf ---- ")
tfidf_vectors = bag.convert_to_tfidf_vectors(dataset.X)
print("tfidf_vectors: ", len(tfidf_vectors))

X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, dataset.y, test_size=test_size)

sf = SimilarityForest(n_trees, fn.cosine_similarity, n_pairs)

sf.fit(X_train, y_train)

sf.predict(X_test)

matrix = sf.get_confusion_matrix(y_test)
print(matrix)