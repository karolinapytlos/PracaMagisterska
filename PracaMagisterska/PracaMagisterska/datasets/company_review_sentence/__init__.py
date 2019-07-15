from ..utils import DatasetLoader, Dataset, get_X_y


#SOURCE: https://dataturks.com/projects/lukef/Company%20Review%20Sentence%20Classification
class CompanyReviewSentenceDataset(DatasetLoader):

    @staticmethod
    def load_data(file_name, name='CompanyReviewSentence'):
        data = []
        with open(CompanyReviewSentenceDataset.get_dataset_file(['company_review_sentence', file_name]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    if sequence is not None and label is not None:
                        data.append((sequence, label))

        X, y = get_X_y(data)
        return Dataset(X, y, name)
