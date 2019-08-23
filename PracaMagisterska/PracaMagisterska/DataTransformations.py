import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformations.n_grams import NGramsTransformer
from transformations.bag_of_words import BagOfWordsTransformer
from transformations.tf_idf import TfIdfTransformer
from datasets.utils import DatasetLoader, Dataset, get_X_y
from datasets.airline_tweets import AirlineTweetsDataset
from datasets.DataType import Type
from DataPreprocessing import Preprocessing

class Transformations:

    def __count_word_in_sequence (word, sequence):
        counter = 0
        for item in sequence:
            if item == word:
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
                words.sort()
            
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
                words.sort()

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




docs = [
("@VirginAmerica plus you've added commercials to the experience... tacky.",	"negative"),
("@VirginAmerica it's really aggressive to blast obnoxious entertainment in your guests' faces &amp; they have little recourse",	"negative"),
("@VirginAmerica and it's a really big bad thing about it",	"negative"),
("@VirginAmerica seriously would pay $30 a flight for seats that didn't have this playing.it's really the only bad thing about flying VA", "negative"),
("@VirginAmerica yes, nearly every time I fly VX this “ear worm” won’t go away :)",	"negative"),
("@virginamerica Well, I didn't…but NOW I DO! :-D",	"positive"),
("@VirginAmerica it was amazing, and arrived an hour early. You're too good to me.", "positive"),
("@VirginAmerica I &lt;3 pretty graphics. so much better than minimal iconography. :D",	"positive"),
("@VirginAmerica This is such a great deal! Already thinking about my 2nd trip to @Australia &amp; I haven't even gone on my 1st trip yet! ;p",	"positive"),
("@VirginAmerica @virginmedia I'm flying your #fabulous #Seductive skies again! U take all the #stress away from travel http://t.co/ahlXHhKiyn",	"positive"),
("@USAirways Flight 496. How are u going to compensate me for sitting on the tarmac for 90+ mins and missing my PHX - BOS connection?",	"negative"),
("@USAirways on top or having to check my bag I had to wait over 30 min for my bag to come out at baggage claim. Thanks for wasting my time",	"negative"),
("@VirginAmerica I flew from NYC to SFO last week and couldn't fully sit in my seat due to two large gentleman on either side of me. HELP!",	"negative"),
("I ❤️ flying @VirginAmerica. ☺️👍",	"positive"),
("@VirginAmerica you know what would be amazingly awesome? BOS-FLL PLEASE!!!!!!! I want to fly with only you.",	"positive"),
("@VirginAmerica why are your first fares in May over three times more than other carriers when all seats are available to select???",	"negative"),
("@VirginAmerica I love this graphic. http://t.co/UT5GrRwAaA",	"positive"),
("@VirginAmerica I love the hipster innovation. You are a feel good brand.",	"positive"),
("@VirginAmerica you guys messed up my seating.. I reserved seating with my friends and you guys gave my seat away ... 😡 I want free internet",	"negative"),
("@VirginAmerica status match program. I applied and it's been three weeks. Called and emailed with no response.",	"negative"),
("@VirginAmerica What happened 2 ur vegan food options?! At least say on ur site so i know I won't be able 2 eat anything for next 6 hrs #fail",	"negative"),
("@VirginAmerica amazing to me that we can't get any cold air from the vents. #VX358 #noair #worstflightever #roasted #SFOtoBOS",	"negative"),
("@VirginAmerica hi! I just bked a cool birthday trip with you, but i can't add my elevate no. cause i entered my middle name during Flight Booking Problems 😢",	"negative")]


docs = Preprocessing.deleteUrls(docs)
docs = Preprocessing.deleteHtmlEntities(docs)
docs = Preprocessing.deleteHtmlTags(docs)
docs = Preprocessing.deletePunctuation(docs)
docs = Preprocessing.deleteMultipleSpaces(docs)

#docs_text = AirlineTweetsDataset.load_data(data_type=Type.text, clear_data=True, data_sample=True, n_samples=10, n_samples_for_class=False)

docs2 = [item[0] for item in docs]
print(docs2)
#bow = BagOfWordsTransformer()
#ft = bow.fit_transform()
#print("ft: ", len(ft))


#docs_vec = AirlineTweetsDataset.load_data(data_type=Type.vector, clear_data=True, data_sample=True, n_samples=10, n_samples_for_class=False)

X, y = get_X_y(docs)
vbow = Transformations.convert_to_bow_vectors(X)
print("vbow: ", len(vbow))