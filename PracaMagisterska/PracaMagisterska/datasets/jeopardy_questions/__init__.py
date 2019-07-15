from ..utils import DatasetLoader, Dataset, get_X_y


#SOURCE: https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/
class JeopardyQuestionsDataset(DatasetLoader):

    @staticmethod
    def load_data(file_name, name='JeopardyQuestions'):
        data = []
        with open(JeopardyQuestionsDataset.get_dataset_file(['jeopardy_questions', file_name]), "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in lines:
                if "\t" in line:
                    sequence, label = line.replace('\n', '').split('\t')
                    if sequence is not None and label is not None:
                        data.append((sequence, label))

        X, y = get_X_y(data)
        return Dataset(X, y, name)