from sklearn.model_selection import train_test_split
from DataTransformations import Transformations
from SimilarityForestClassifier import SimilarityForest

class Initiate:
    def __init__ (self, n_trees, n_object_pairs, similarity_function, test_dataset_size, dataset_type,
                  clear_data, data_sample, n_samples, n_samples_for_class):
        self.__n_trees = n_trees
        self.__n_object_pairs = n_object_pairs
        self.__similarity_function = similarity_function
        self.__test_dataset_size = test_dataset_size
        self.__similarity_forest = SimilarityForest(self.__n_trees, self.__similarity_function, self.__n_object_pairs)
        self.__dataset = dataset_type.load_data(clear_data, data_sample, n_samples, n_samples_for_class)

    def run_bag_of_words (self):
        print(" \n---- BAG OF WORDS ---- ")
        bag_vectors = Transformations.convert_to_bow_vectors(self.__dataset.X)
        X_train, X_test, y_train, y_test = train_test_split(bag_vectors, self.__dataset.y, test_size=self.__test_dataset_size)
        self.__similarity_forest.fit(X_train, y_train)
        self.__similarity_forest.predict(X_test)
        print(self.__similarity_forest.get_confusion_matrix(y_test))

    def run_tf_idf (self):
        print(" \n---- TF-IDF ---- ")
        tfidf_vectors = Transformations.convert_to_tfidf_vectors(self.__dataset.X)
        X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, self.__dataset.y, test_size=self.__test_dataset_size)
        self.__similarity_forest.fit(X_train, y_train)
        self.__similarity_forest.predict(X_test)
        print(self.__similarity_forest.get_confusion_matrix(y_test))