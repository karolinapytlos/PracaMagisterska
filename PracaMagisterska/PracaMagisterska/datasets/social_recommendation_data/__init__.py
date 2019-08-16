from ..utils import DatasetLoader, Dataset, get_X_y
from DataPreprocessing import Preprocessing


# SOURCE: https://cseweb.ucsd.edu/~jmcauley/datasets.html
class SocialRecommendationDataset(DatasetLoader):

    @staticmethod
    def load_data(clear_data, data_sample, n_samples, n_samples_for_class, name='SocialRecommendation'):
        data = []
        with open(SocialRecommendationDataset.get_dataset_file(['social_recommendation_data', "recommendation.txt"]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    if sequence is not None and label is not None:
                        if clear_data == True:
                            sequence = Preprocessing.deleteUrls(sequence)
                            sequence = Preprocessing.deleteHtmlEntities(sequence)
                            sequence = Preprocessing.deleteHtmlTags(sequence)
                        data.append((sequence, label))

        if data_sample == True:
            data = SocialRecommendationDataset.sampled_data(data, n_samples, n_samples_for_class)

        X, y = get_X_y(data)
        return Dataset(X, y, name)
