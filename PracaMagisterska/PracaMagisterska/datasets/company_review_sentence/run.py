from ..InitiateDataset import Initiate
from . import CompanyReviewSentenceDataset

class CompanyReviews (Initiate):
    def __init__(self, n_trees, n_object_pairs, similarity_function, test_dataset_size,
                 dataset_type= CompanyReviewSentenceDataset, clear_data=True, data_sample=True, n_samples=100, n_samples_for_class=True):
        print(" \n---- COMPANY REVIEWS ---- ")
        return super().__init__(n_trees, n_object_pairs, similarity_function, test_dataset_size,
                                dataset_type, clear_data, data_sample, n_samples, n_samples_for_class)