from ..InitiateDataset import Initiate
from . import SocialRecommendationDataset
from ..DataType import Type

class SocialRecommendation (Initiate):
    def __init__(self, n_trees, n_object_pairs, similarity_function, test_dataset_size,
                 dataset_type=SocialRecommendationDataset, data_type=Type.vector, clear_data=True, data_sample=True, n_samples=50, n_samples_for_class=True):
        print(" \n ---- SOCIAL RECOMMENDATION ---- ")
        print(" ----", similarity_function.__name__, "---- ")
        return super().__init__(n_trees, n_object_pairs, similarity_function, test_dataset_size,
                                dataset_type, data_type, clear_data, data_sample, n_samples, n_samples_for_class)