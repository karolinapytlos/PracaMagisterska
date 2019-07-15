from ..utils import DatasetLoader, Dataset, get_X_y


#SOURCE: https://dataturks.com/projects/markus.radtke/classify%20auto
class AutomotiveDataset(DatasetLoader):

    @staticmethod
    def load_data(file_name, name='Automotive'):
        data = []
        with open(AutomotiveDataset.get_dataset_file(['automotive', file_name]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    if sequence is not None and label is not None:
                        data.append((sequence, label))

        X, y = get_X_y(data)
        return Dataset(X, y, name)
