from DataConverter import Converter

# Twitter US Airline Sentiment
#data = Converter.loadCSV("D:\Zbiory danych\Twitter US Airline Sentiment\Tweets.csv", ",", "text", "airline_sentiment", ["positive", "negative"])
#if data is not None:
#    print("data: ", len(data))
#    result = Converter.save_file(".\\datasets\\airline_tweets\\tweets.txt", data, '\t')
#    print("File tweets.csv was saved: ", result)


# Social Recommendation Data
#data = Converter.load_social_recommendation_data("D:\Zbiory danych\Social Recommendation Data\epinions_data\epinions.json", "review", "stars", [1, 5])
#if data is not None:
#    print("data: ", len(data))
#    result = Converter.save_file(".\\datasets\\social_recommendation_data\\recommendation.txt", data, '\t')
#    print("File recommendation.txt was saved: ", result)


# SMS Spam Collection
#data = Converter.loadTXT("D:\Zbiory danych\SMS Spam Collection\SMSSpamCollection.txt", '\t', 1, 0)
#if data is not None:
#    print("data: ", len(data))
#    result = Converter.save_file(".\\datasets\\sms_spam_collection\\sms_spam.txt", data, '\t')
#    print("File sms_spam.txt was saved: ", result)


# Large Movie Review Dataset
#data_train_pos = Converter.load_large_movie_review_dataset(r"D:\Zbiory danych\Large Movie Review Dataset\aclImdb\train\pos", "pos")
#if data_train_pos is not None:
#    print("data_train_pos: ", len(data_train_pos))

#data_train_neg = Converter.load_large_movie_review_dataset(r"D:\Zbiory danych\Large Movie Review Dataset\aclImdb\train\neg", "neg")
#if data_train_neg is not None:
#    print("data_train_neg: ", len(data_train_neg))

#if data_train_pos is not None and data_train_neg is not None:
#    data_train = Converter.merge_datasets(data_train_pos, data_train_neg)
#    print("data_train: ", len(data_train))

#data_test_pos = Converter.load_large_movie_review_dataset(r"D:\Zbiory danych\Large Movie Review Dataset\aclImdb\test\pos", "pos")
#if data_test_pos is not None:
#    print("data_test_pos: ", len(data_test_pos))

#data_test_neg = Converter.load_large_movie_review_dataset(r"D:\Zbiory danych\Large Movie Review Dataset\aclImdb\test\neg", "neg")
#if data_test_neg is not None:
#    print("data_test_neg: ", len(data_test_neg))

#if data_test_pos is not None and data_test_neg is not None:
#    data_test = Converter.merge_datasets(data_test_pos, data_test_neg)
#    print("data_test: ", len(data_test))

#if data_train is not None and data_test is not None:
#    data = Converter.merge_datasets(data_train, data_test)
#    if data is not None:
#        print("data: ", len(data))
#        result = Converter.save_file(".\\datasets\\movie_reviews\\reviews.txt", data, '\t')
#        print("File reviews.txt was saved: ", result)


# Jeopardy! Questions
#data = Converter.loadJSON(r"D:\Zbiory danych\Jeopardy! Questions\JEOPARDY_QUESTIONS1.json", "question", "category", ["SPORTS", "HISTORY"])
#if data is not None:
#    print("data: ", len(data))
#    result = Converter.save_file(".\\datasets\\jeopardy_questions\\questions.txt", data, '\t')
#    print("File questions.txt was saved: ", result)


# Clothing Fit Data - Rent The Runway
# w edytorze tekstowym zamień "}" na "}," , dodaj na początku pliku przed pierwszym "{" znak "[" , przy ostatnim znaku "}," usuń "," i dodaj "]" , zapisz plik
#data = Converter.loadJSON(r"D:\Zbiory danych\Clothing Fit Data\renttherunway_final_data_copy.json", "review_text", "rented for", ["everyday", "work"])
#if data is not None:
#    print("data: ", len(data))
#    result = Converter.save_file(".\\datasets\\clothing_fit_data\\renttherunway.txt", data, '\t')
#    print("File renttherunway.txt was saved: ", result)


# The Blog Authorship Corpus - 25.Student
# z folderu blog wybierz tylko pliki 25.Student
#data = Converter.load_blog_authorship_corpus(r"D:\Zbiory danych\The Blog Authorship Corpus\25_Student")
#if data is not None:
#    print("data: ", len(data))
#    result = Converter.save_file(".\\datasets\\blog_authorship_corpus\\blog.txt", data, '\t')
#    print("File blog.txt was saved: ", result)


# Automotive
#data = Converter.load_automotive(r"D:\Zbiory danych\Automotive\classify auto.json")
#if data is not None:
#    print("data: ", len(data))
#    result = Converter.save_file(".\\datasets\\automotive\\classify_auto.txt", data, '\t')
#    print("File classify_auto.txt was saved: ", result)


# Company Review Sentence
#data = Converter.load_company_review_sentence(r"D:\Zbiory danych\Company Review Sentence\Company Review Sentence Classification.json")
#if data is not None:
#    print("data: ", len(data))
#    result = Converter.save_file(".\\datasets\\company_review_sentence\\company_sentence.txt", data, '\t')
#    print("File company_sentence.txt was saved: ", result)


# IMDB Movie Genre
#data = Converter.load_imdb_movie_genre(r"D:\Zbiory danych\IMBD Movie Genre\IMDB Movie Genre Dataset.json")
#if data is not None:
#    print("data: ", len(data))
#    result = Converter.save_file(".\\datasets\\imdb_movie_genre\\imdb_movies.txt", data, '\t')
#    print("File imdb_movies.txt was saved: ", result)


# Indian Politics News 2018
#data = Converter.loadCSV(r"D:\Zbiory danych\Indian Politics News 2018\politics18.csv", ',', "content", "author", ["The Quint", "PTI"])
#if data is not None:
#    print("data: ", len(data))
#    result = Converter.save_file(".\\datasets\\indian_politics_news_2018\\indian_politics_news.txt", data, '\t')
#    print("File indian_politics_news.txt was saved: ", result)


# E-commerce Reviews
#data = Converter.loadCSV(r"D:\Zbiory danych\\Ecommerce review\\Womens Clothing E-Commerce Reviews.csv", ',', "Review Text", "Class Name", ["Dresses", "Knits"])
#if data is not None:
#    print("data: ", len(data))
#    result = Converter.save_file(".\\datasets\\ecommerce_reviews\\ecommerce_reviews.txt", data, '\t')
#    print("File ecommerce_reviews.txt was saved: ", result)