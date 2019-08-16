from ..utils import DatasetLoader, Dataset, get_X_y
from DataPreprocessing import Preprocessing
from ..DataType import Type


# SOURCE: https://www.kaggle.com/kavita5/review_ecommerce
class EcommerceReviewsDataset(DatasetLoader):

    @staticmethod
    def load_data (data_type, clear_data, data_sample, n_samples, n_samples_for_class, name='EcommerceReviews'):
        data = []
        with open(EcommerceReviewsDataset.get_dataset_file(['ecommerce_reviews', "ecommerce_reviews.txt"]), "r", encoding="utf8") as f:
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
            data = EcommerceReviewsDataset.sampled_data(data, n_samples, n_samples_for_class)

        if data_type == Type.text:
            X, y = EcommerceReviewsDataset.get_text_data(data)
            return Dataset(X, y, name)

        X, y = get_X_y(data)
        return Dataset(X, y, name)
