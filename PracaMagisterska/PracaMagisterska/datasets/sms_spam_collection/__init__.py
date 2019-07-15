from ..utils import DatasetLoader, Dataset, get_X_y


#SOURCE: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
class SmsSpamCollectionDataset(DatasetLoader):

    @staticmethod
    def load_data(file_name, name='SmsSpamCollection'):
        data = []
        with open(SmsSpamCollectionDataset.get_dataset_file(['sms_spam_collection', file_name]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    if sequence is not None and label is not None:
                        data.append((sequence, label))

        X, y = get_X_y(data)
        return Dataset(X, y, name)