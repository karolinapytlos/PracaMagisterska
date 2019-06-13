import numpy as np
import os

class HeartDataset:

    def convert_file (file_path):
        base_path = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(base_path, file_path)

        data = []
        labels = []

        columns = [1,2,3,4,5,6,7,8,9,10,11,12,13]
        class_index = 0
        delimiter = ':'

        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '').split(' ')
                if line[len(line) -1] == '':
                    line.pop()

                label = int(line[class_index])
                del line[class_index]

                vector = []
                last_index = 0
                counter = 0
                while counter != len(columns):
                    try:
                        item = line[counter].split(':')
                    except IndexError:
                        break

                    column = int(item[0])
                    if (column in columns) and ((column - 1) == last_index):
                        vector.append(float(item[1]))
                        last_index = column
                        counter = counter + 1
                    else:
                        vector.append(0)
                        last_index = last_index + 1

                if len(vector) > 0:
                    data.append(np.array(vector))
                    labels.append(label)

        return data, labels