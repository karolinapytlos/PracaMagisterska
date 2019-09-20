
if __name__ == "__main__":

    import Functions as fn
    from datasets.DataType import Type

    from datasets.airline_tweets.run import AirlineTweets
    from datasets.automotive.run import Automotive
    from datasets.blog_authorship_corpus.run import BlogAuthorshipCorpus
    from datasets.clothing_fit_data.run import ClothingFit
    from datasets.company_review_sentence.run import CompanyReviews
    from datasets.ecommerce_reviews.run import EcommerceReviews
    from datasets.imdb_movie_genre.run import ImdbMoviesGenre
    from datasets.indian_politics_news_2018.run import IndianPoliticsNews
    from datasets.jeopardy_questions.run import JeopardyQuestions
    from datasets.movie_reviews.run import MovieReviews
    from datasets.sms_spam_collection.run import SmsSpam
    from datasets.social_recommendation_data.run import SocialRecommendation

    n_trees = 10
    n_pairs = 3
    similarity_functions = [fn.cosine_distance, fn.longest_common_subsequence, fn.euclidean_distance, fn.levenshtein_distance]
    test_size = 0.20

    print(" --- START --- ")

    airline_tweets = AirlineTweets(n_trees, n_pairs, similarity_functions[0], test_size)
    airline_tweets.run_bag_of_words()
    #airline_tweets_text = AirlineTweets(n_trees, n_pairs, similarity_functions[3], test_size, data_type=Type.text)
    #airline_tweets_text.run_text()
    #airline_tweets_ngrams = AirlineTweets(n_trees, n_pairs, similarity_functions[0], test_size)
    #airline_tweets_ngrams.run_n_grams(2)


    #automotive = Automotive(n_trees, n_pairs, similarity_functions[0], test_size)
    #automotive.run_bag_of_words()
    #automotive.run_tf_idf()
    #automotive_text = Automotive(n_trees, n_pairs, similarity_functions[1], test_size, data_type=Type.text)
    #automotive_text.run_text()


    #blog_authorship_corpus = BlogAuthorshipCorpus(n_trees, n_pairs, similarity_functions[0], test_size)
    #blog_authorship_corpus.run_bag_of_words()
    #blog_authorship_corpus_text = BlogAuthorshipCorpus(n_trees, n_pairs, similarity_functions[1], test_size, data_type=Type.text)
    #blog_authorship_corpus_text.run_text()


    #clothing_fit = ClothingFit(n_trees, n_pairs, similarity_functions[0], test_size)
    #clothing_fit.run_bag_of_words()
    #clothing_fit_text = ClothingFit(n_trees, n_pairs, similarity_functions[1], test_size, data_type=Type.text)
    #clothing_fit_text.run_text()


    #company_reviews = CompanyReviews(n_trees, n_pairs, similarity_functions[0], test_size)
    #company_reviews.run_bag_of_words()
    #company_reviews_text = CompanyReviews(n_trees, n_pairs, similarity_functions[1], test_size, data_type=Type.text)
    #company_reviews_text.run_text()


    #ecommerce_reviews = EcommerceReviews(n_trees, n_pairs, similarity_functions[0], test_size)
    #ecommerce_reviews.run_bag_of_words()
    #ecommerce_reviews.run_tf_idf()
    #ecommerce_reviews_text = EcommerceReviews(n_trees, n_pairs, similarity_functions[1], test_size, data_type=Type.text)
    #ecommerce_reviews_text.run_text()


    #imdb_movies_genre = ImdbMoviesGenre(n_trees, n_pairs, similarity_functions[0], test_size)
    #imdb_movies_genre.run_bag_of_words()
    #imdb_movies_genre.run_tf_idf()
    #imdb_movies_genre_text = ImdbMoviesGenre(n_trees, n_pairs, similarity_functions[1], test_size, data_type=Type.text)
    #imdb_movies_genre_text.run_text()


    #indian_politics_news = IndianPoliticsNews(n_trees, n_pairs, similarity_functions[0], test_size)
    #indian_politics_news.run_bag_of_words()
    #indian_politics_news.run_tf_idf()
    #indian_politics_news_text = IndianPoliticsNews(n_trees, n_pairs, similarity_functions[1], test_size, data_type=Type.text)
    #indian_politics_news_text.run_text()


    #jeopardy_questions = JeopardyQuestions(n_trees, n_pairs, similarity_functions[0], test_size)
    #jeopardy_questions.run_bag_of_words()
    #jeopardy_questions.run_tf_idf()
    #jeopardy_questions_text = JeopardyQuestions(n_trees, n_pairs, similarity_functions[1], test_size, data_type=Type.text)
    #jeopardy_questions_text.run_text()


    #movie_reviews = MovieReviews(n_trees, n_pairs, similarity_functions[0], test_size)
    #movie_reviews.run_bag_of_words()
    #movie_reviews_text = MovieReviews(n_trees, n_pairs, similarity_functions[1], test_size, data_type=Type.text)
    #movie_reviews_text.run_text()


    #sms_spam = SmsSpam(n_trees, n_pairs, similarity_functions[0], test_size)
    #sms_spam.run_bag_of_words()
    #sms_spam_text = SmsSpam(n_trees, n_pairs, similarity_functions[1], test_size, data_type=Type.text)
    #sms_spam_text.run_text()


    #social_recommendation = SocialRecommendation(n_trees, n_pairs, similarity_functions[0], test_size)
    #social_recommendation.run_bag_of_words()
    #social_recommendation.run_tf_idf()
    #social_recommendation_text = SocialRecommendation(n_trees, n_pairs, similarity_functions[1], test_size, data_type=Type.text)
    #social_recommendation_text.run_text()

    print(" --- END --- ")