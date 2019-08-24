import math

class Transformations:

    def __count_word_in_sequence (word, sequence):
        counter = 0
        for item in sequence:
            if item == word:
                counter = counter + 1

        return counter


    def __count_ngram_in_sequence (ngram, sequence, range):
        counter = 0
        for index, item in enumerate(sequence):
            subset = sequence[index : range + index]
            if len(subset) == range:
                if subset == ngram:
                    counter = counter + 1

        return counter


    def __check_word_in_sequence (word, sequence):
        for item in sequence:
            if item == word:
                return True

        return False


    def __tf (word, sequence):
        return Transformations.__count_word_in_sequence(word, sequence) / len(sequence)


    def __idf (word_occurrence, len_dataset):
        return math.log10(len_dataset / word_occurrence)

    @staticmethod
    def convert_to_bow_vectors (dataset):
        vectors = []
        if type(dataset) is list:
            # 1. Get all unique words
            words = []
            for sequence in dataset:
                for word in sequence:
                    if word not in words:
                        words.append(word)

            if len(words) > 0:
                # 2. Count words in sequence
                for sequence in dataset:
                    vector = []
                    for word in words:
                        count = Transformations.__count_word_in_sequence(word, sequence)
                        vector.append(count)
                    vectors.append(vector)

        return vectors

    @staticmethod
    def convert_to_tfidf_vectors (dataset):
        vectors = []
        if type(dataset) is list:
            # 1. Get all unique words
            words = []
            for sequence in dataset:
                for word in sequence:
                    if word not in words:
                        words.append(word)

            if len(words) > 0:
                # 2. Count words in documents
                word_in_document = []
                for word in words:
                    counter = 0
                    for sequence in dataset:
                        if Transformations.__check_word_in_sequence(word, sequence):
                            counter = counter + 1
                    word_in_document.append((word, counter))

            # 3. Calculate TD-IDF
            if len(word_in_document) > 0:
                for sequence in dataset:
                    vector = []
                    for word, occurrence in word_in_document:
                        tf = Transformations.__tf(word, sequence)
                        idf = Transformations.__idf(occurrence, len(dataset))
                        vector.append(tf * idf)

                    vectors.append(vector)

        return vectors

    @staticmethod
    def convert_to_ngrams_vectors (dataset, range=2):
        if range < 2:
            print("Use convert_to_bow_vectors function.")
            return

        vectors = []
        if type(dataset) == list:
            if len(dataset) > 0:
                # 1. Create vector of n-grams
                ngrams_vector = []
                for sequence in dataset:
                    ngrams_index = 0
                    gram = []
                    for index, item in enumerate(sequence):
                        gram = sequence[index : range + index]
                        if len(gram) == range:
                            if gram not in ngrams_vector:
                                ngrams_vector.append(gram)

                # 2. Count n-grams in sequences
                if len(ngrams_vector) > 0:
                    for sequence in dataset:
                        vector = []
                        for ngram in ngrams_vector:
                            count = Transformations.__count_ngram_in_sequence(ngram, sequence, range)
                            vector.append(count)            
                        vectors.append(vector)
        return vectors