from ..InitiateDataset import Initiate
from . import IndianPoliticsNews2018Dataset

class IndianPoliticsNews (Initiate):
    def __init__(self, n_trees, n_object_pairs, similarity_function, test_dataset_size,
                 dataset_type=IndianPoliticsNews2018Dataset, clear_data=True, data_sample=True, n_samples=100, n_samples_for_class=True):
        print(" \n---- INDIAN POLITICS NEWS ---- ")
        return super().__init__(n_trees, n_object_pairs, similarity_function, test_dataset_size,
                                dataset_type, clear_data, data_sample, n_samples, n_samples_for_class)