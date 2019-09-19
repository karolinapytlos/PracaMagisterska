from ..utils import DatasetLoader, Dataset, get_X_y
from DataPreprocessing import Preprocessing
from ..DataType import Type


# SOURCE: http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
class BlogAuthorshipCorpusDataset(DatasetLoader):

    @staticmethod
    def load_data (data_type, clear_data, data_sample=False, n_samples=0, n_samples_for_class=False, name='Blog authorship corpus'):
        data = []
        with open(BlogAuthorshipCorpusDataset.get_dataset_file(['blog_authorship_corpus', "blog.txt"]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    # change categorical value to numeric
                    if label == "female":
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
            data = BlogAuthorshipCorpusDataset.sampled_data(data, n_samples, n_samples_for_class)

        if data_type == Type.text:
            X, y = BlogAuthorshipCorpusDataset.get_text_data(data)
            return Dataset(X, y, name)

        X, y = get_X_y(data)
        return Dataset(X, y, name)