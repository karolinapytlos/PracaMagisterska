from ..utils import DatasetLoader, Dataset, get_X_y


#SOURCE: https://www.kaggle.com/xenomorph/indian-politics-news-2018
class IndianPoliticsNews2018Dataset(DatasetLoader):

    @staticmethod
    def load_data(file_name, name='IndianPoliticsNews2018'):
        data = []
        with open(IndianPoliticsNews2018Dataset.get_dataset_file(['indian_politics_news_2018', file_name]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    if sequence is not None and label is not None:
                        data.append((sequence, label))

        X, y = get_X_y(data)
        return Dataset(X, y, name)
