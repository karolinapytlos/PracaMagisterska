from ..InitiateDataset import Initiate
from . import JeopardyQuestionsDataset
from ..DataType import Type

class JeopardyQuestions (Initiate):
    def __init__(self, n_trees, n_object_pairs, similarity_function, test_dataset_size,
                 dataset_type=JeopardyQuestionsDataset, data_type=Type.vector, clear_data=True, data_sample=True, n_samples=50, n_samples_for_class=True):
        print(" \n---- JEOPARDY QUESTIONS ---- ")
        print(" ----", similarity_function.__name__, "---- ")
        return super().__init__(n_trees, n_object_pairs, similarity_function, test_dataset_size,
                                dataset_type, data_type, clear_data, data_sample, n_samples, n_samples_for_class)
