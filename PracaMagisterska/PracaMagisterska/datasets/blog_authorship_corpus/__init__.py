from ..utils import DatasetLoader, Dataset, get_X_y


#SOURCE: http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
class BlogAuthorshipCorpusDataset(DatasetLoader):

    @staticmethod
    def load_data(file_name, name='BlogAuthorshipCorpus'):
        data = []
        with open(BlogAuthorshipCorpusDataset.get_dataset_file(['blog_authorship_corpus', file_name]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    if sequence is not None and label is not None:
                        data.append((sequence, label))

        X, y = get_X_y(data)
        return Dataset(X, y, name)