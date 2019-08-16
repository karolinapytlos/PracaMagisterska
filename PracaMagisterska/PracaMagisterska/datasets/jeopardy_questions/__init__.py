from ..utils import DatasetLoader, Dataset, get_X_y
from DataPreprocessing import Preprocessing


# SOURCE: https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/
class JeopardyQuestionsDataset(DatasetLoader):

    @staticmethod
    def load_data (clear_data, data_sample, n_samples, n_samples_for_class, name='JeopardyQuestions'):
        data = []
        with open(JeopardyQuestionsDataset.get_dataset_file(['jeopardy_questions', "questions.txt"]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    if sequence is not None and label is not None:
                        if clear_data == True:
                            sequence = Preprocessing.deleteUrls(sequence)
                            sequence = Preprocessing.deleteHtmlEntities(sequence)
                            sequence = Preprocessing.deleteHtmlTags(sequence)
                        data.append((sequence, label))

        if data_sample == True:
            data = JeopardyQuestionsDataset.sampled_data(data, n_samples, n_samples_for_class)

        X, y = get_X_y(data)
        return Dataset(X, y, name)