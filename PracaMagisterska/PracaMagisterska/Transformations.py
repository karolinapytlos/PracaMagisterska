import math


class BagOfWords:

    def __count_word_in_sequence (self, word, sequence):
        counter = 0
        for item in sequence:
            if item == word:
                counter = counter + 1

        return counter


    def __check_word_in_sequence (self, word, sequence):
        for item in sequence:
            if item == word:
                return True

        return False


    def __tf (self, word, sequence):
        return self.__count_word_in_sequence(word, sequence) / len(sequence)


    def __idf (self, word_occurrence, len_dataset):
        return math.log10(len_dataset / word_occurrence)


    def convert_to_bow_vectors (self, dataset):
        vectors = []
        if type(dataset) is list:
            # 1. Get all unique words
            words = []
            for sequence in dataset:
                for word in sequence:
                    if word not in words:
                        words.append(word)

            if len(words) > 0:
                words.sort()
            
            # 2. Count words in sequence
            for sequence in dataset:
                vector = []
                for word in words:
                    count = self.__count_word_in_sequence(word, sequence)
                    vector.append(count)

                vectors.append(vector)

        return vectors


    def convert_to_tfidf_vectors (self, dataset):
        vectors = []
        if type(dataset) is list:
            # 1. Get all unique words
            words = []
            for sequence in dataset:
                for word in sequence:
                    if word not in words:
                        words.append(word)

            if len(words) > 0:
                words.sort()

            # 2. Count words in documents
            word_in_document = []
            for word in words:
                counter = 0
                for sequence in dataset:
                    if self.__check_word_in_sequence(word, sequence):
                        counter = counter + 1

                word_in_document.append((word, counter))

            # 3. Calculate TD-IDF
            if len(word_in_document) > 0:
                for sequence in dataset:
                    vector = []
                    for word, occurrence in word_in_document:
                        tf = self.__tf(word, sequence)
                        idf = self.__idf(occurrence, len(dataset))
                        vector.append(tf * idf)

                    vectors.append(vector)

        return vectors