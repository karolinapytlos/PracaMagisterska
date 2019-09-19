from ..utils import DatasetLoader, Dataset, get_X_y
from DataPreprocessing import Preprocessing
from ..DataType import Type


# SOURCE: https://dataturks.com/projects/GajendraDadheech3/IMDB%20Movie%20Genre%20Dataset
class IMDBMovieGenreDataset(DatasetLoader):

    @staticmethod
    def load_data (data_type, clear_data, data_sample=False, n_samples=0, n_samples_for_class=False, name='IMDB movie genre'):
        data = []
        with open(IMDBMovieGenreDataset.get_dataset_file(['imdb_movie_genre', "imdb_movies.txt"]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    # change categorical value to numeric
                    if label == "Action":
                        label = 1
                    else:
                        label = 0

                    if sequence is not None and label is not None:
                        if clear_data == True:
                            sequence = Preprocessing.deleteUrls(sequence)
                            sequence = Preprocessing.deleteHtmlEntities(sequence)
                            sequence = Preprocessing.deleteHtmlTags(sequence)
                            sequence = Preprocessing.deletePunctuation(sequence)
                            sequence = Preprocessing.deleteMultipleSpaces(sequence)
                        data.append((sequence.strip(), label))

        if data_sample == True:
            data = IMDBMovieGenreDataset.sampled_data(data, n_samples, n_samples_for_class)

        if data_type == Type.text:
            X, y = IMDBMovieGenreDataset.get_text_data(data)
            return Dataset(X, y, name)

        X, y = get_X_y(data)
        return Dataset(X, y, name)