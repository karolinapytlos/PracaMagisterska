from ..utils import DatasetLoader, Dataset, get_X_y
from DataPreprocessing import Preprocessing


# SOURCE: https://dataturks.com/projects/markus.radtke/classify%20auto
class AutomotiveDataset(DatasetLoader):

    @staticmethod
    def load_data (clear_data, data_sample, n_samples, n_samples_for_class, name='Automotive'):
        data = []
        with open(AutomotiveDataset.get_dataset_file(['automotive', "classify_auto.txt"]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    if sequence is not None and label is not None:
                        if clear_data == True:
                            sequence = Preprocessing.deleteUrls(sequence)
                            sequence = Preprocessing.deleteHtmlEntities(sequence)
                            sequence = Preprocessing.deleteHtmlTags(sequence)
                        data.append((sequence.strip(), label))

        if data_sample == True:
            data = AutomotiveDataset.sampled_data(data, n_samples, n_samples_for_class)

        X, y = get_X_y(data)
        return Dataset(X, y, name)
