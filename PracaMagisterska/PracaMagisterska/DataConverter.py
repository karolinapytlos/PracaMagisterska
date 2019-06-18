import random as rd
import csv
import json
import os
from bs4 import BeautifulSoup
import re

class Converter:

    def loadCSV (file_path, columns_delimiter, sequence_column_name, label_column_name, labels_value):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            sequenceIndex = 0
            labelIndex = 0
            for index, name in enumerate(header):
                if name == sequence_column_name:
                    sequenceIndex = index
                if name == label_column_name:
                    labelIndex = index

            for row in reader:
                itemSequence = None
                itemLabel = None
                for index, value in enumerate(row):
                    if sequenceIndex == index:
                        itemSequence = value
                    if labelIndex == index:
                        if value in labels_value:
                            itemLabel = value
                    if itemSequence is not None and itemLabel is not None:
                        data.append((itemSequence, itemLabel))
                        break
        f.close()

        return data


    def load_social_recommendation_data (file_path, sequence_property_name, label_property_name, labels_value):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                line = line.replace("\\x08", '')

                # examples: "don''t", "don't"
                apostrophes = re.findall("\"+[\w\']{1,}\"", line)
                if apostrophes is not None:
                    for match in apostrophes:
                        match = match.replace("\'", "\'\'")
                        line = re.sub("\"+[\w\']{1,}\"", match, line)

                # exmaples: '"Rental"'
                apostrophes = re.findall("\'\"+[\w\']{1,}\"\'", line)
                if apostrophes is not None:
                    for match in apostrophes:
                        match = match.replace("\'\"", "\'").replace("\"\'", "\'")
                        line = re.sub("\'\"+[\w\']{1,}\"\'", match, line)

                # exmaples: '42"'
                apostrophes = re.findall("\'+[\w\']{1,}\"\'", line)
                if apostrophes is not None:
                    for match in apostrophes:
                        match = match.replace("\"\'", "\'")
                        line = re.sub("\'+[\w\']{1,}\"\'", match, line)

                # exmaples: '113.00"4day"'
                apostrophes = re.findall("\'+[\w\\d.']{1,}\"[\w\\d.']+\"\'", line)
                if apostrophes is not None:
                    for match in apostrophes:
                        match = match.replace("\"\'", "\'").replace("\"", ' ')
                        line = re.sub("\'+[\w\\d.']{1,}\"[\w\\d.']+\"\'", match, line)

                # exmaples: "'92CDN24000"
                apostrophes = re.findall("\"\'+[\w\\d.']{1,}\"", line)
                if apostrophes is not None:
                    for match in apostrophes:
                        match = match.replace("\'", '').replace("\"", "\'")
                        line = re.sub("\"\'+[\w\\d.']{1,}\"", match, line)
                
                line = line.replace("'", '\"').replace('\"\"', "\'")
                json_line = json.loads(line)
                sequenceValue = json_line[sequence_property_name]
                labelValue = json_line[label_property_name]
                if sequenceValue is not None and labelValue is not None:
                    if labelValue in labels_value:
                        data.append((sequenceValue.strip(), labelValue))

        return data


    def loadJSON (file_path, sequence_property_name, label_property_name, labels):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = json.load(f)
            for item in reader:
                if sequence_property_name in item and label_property_name in item:
                    sequenceValue = item[sequence_property_name]
                    labelValue = item[label_property_name]
                    if sequenceValue.startswith("\'"):
                        sequenceValue = sequenceValue[1:]
                    if sequenceValue.endswith("\'"):
                        sequenceValue = sequenceValue[:-1]
                    if sequenceValue is not None and labelValue is not None:
                        if labelValue in labels:
                            data.append((sequenceValue, labelValue))

        return data


    def loadTXT (file_path, delimiter, sequence_index, label_index):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('  ', ' ').replace('\n', '').split(delimiter)
                if len(line) == 2:
                    data.append((line[sequence_index], line[label_index]))

        return data


    def load_large_movie_review_dataset(dictionary_path, label):
        data = []
        files = os.listdir(dictionary_path)
        for filename in files:
            with open(dictionary_path +"\\"+ filename, "r", encoding="utf-8") as f:
                line = f.read()
                data.append((line, label))
                print("File ", filename, " has been readed.")

        return data


    def load_blog_authorship_corpus (dictionary_path):
        data = []
        files = os.listdir(dictionary_path)
        for filename in files:
            file_path = dictionary_path +"\\"+ filename
            print("File ", filename)
            firstIndex = filename.find('.')
            if firstIndex > 0:
                subStr = filename[firstIndex +1:]
                label = subStr[: subStr.find('.')].strip()
                if len(label) > 0:
                    file_data = Converter.loadXML(file_path, "post", label)
                    if file_data is not None:
                        data.extend(file_data)
                        print("File ", filename, " has been readed.")
        return data


    def loadXML (file_path, sequence_element_name, label):
        data = []
        with open(file_path, "rb") as f:
            soup = BeautifulSoup(f, "xml")
            for tag in soup.find_all(sequence_element_name):
                data.append((tag.string.replace('  ', ' ').replace('\n', '').strip(), label))

        return data


    def merge_datasets (*args):
        dataset = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                dataset = dataset + arg

        if len(dataset) > 0:
            return rd.sample(dataset, len(dataset))

        return dataset


    def save_file (file_path, dataset, data_delimiter):
        if isinstance(dataset, (list, tuple)):
            if len(dataset) > 0:
                new_file = open(file_path, "w+", encoding="utf-8")
                for item in dataset:
                    new_file.write(item[0] + data_delimiter + str(item[1]) + "\n")
                
                new_file.close()
                return True

        return False