from ..utils import DatasetLoader, Dataset, get_X_y


#SOURCE: https://cseweb.ucsd.edu/~jmcauley/datasets.html
class SocialRecommendationDataset(DatasetLoader):

    @staticmethod
    def load_data(file_name, name='SocialRecommendation'):
        data = []
        with open(SocialRecommendationDataset.get_dataset_file(['social_recommendation_data', file_name]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    if sequence is not None and label is not None:
                        data.append((sequence, label))

        X, y = get_X_y(data)
        return Dataset(X, y, name)
