from ..utils import DatasetLoader, Dataset, get_X_y


#SOURCE: https://dataturks.com/projects/GajendraDadheech3/IMDB%20Movie%20Genre%20Dataset
class IMDBMovieGenreDataset(DatasetLoader):

    @staticmethod
    def load_data(file_name, name='IMDBMovieGenre'):
        data = []
        with open(IMDBMovieGenreDataset.get_dataset_file(['imdb_movie_genre', file_name]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    if sequence is not None and label is not None:
                        data.append((sequence, label))

        X, y = get_X_y(data)
        return Dataset(X, y, name)
