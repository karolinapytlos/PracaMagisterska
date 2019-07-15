from ..utils import DatasetLoader, Dataset, get_X_y


#SOURCE: https://cseweb.ucsd.edu/~jmcauley/datasets.html
class ClothingFitDataset(DatasetLoader):

    @staticmethod
    def load_data(file_name, name='ClothingFit'):
        data = []
        with open(ClothingFitDataset.get_dataset_file(['clothing_fit_data', file_name]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    if sequence is not None and label is not None:
                        data.append((sequence, label))

        X, y = get_X_y(data)
        return Dataset(X, y, name)