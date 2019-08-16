from ..utils import DatasetLoader, Dataset, get_X_y
from DataPreprocessing import Preprocessing


# SOURCE: http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
class BlogAuthorshipCorpusDataset(DatasetLoader):

    @staticmethod
    def load_data (clear_data, data_sample, n_samples, n_samples_for_class, name='BlogAuthorshipCorpus'):
        data = []
        with open(BlogAuthorshipCorpusDataset.get_dataset_file(['blog_authorship_corpus', "blog.txt"]), "r", encoding="utf8") as f:
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
            data = BlogAuthorshipCorpusDataset.sampled_data(data, n_samples, n_samples_for_class)

        X, y = get_X_y(data)
        return Dataset(X, y, name)