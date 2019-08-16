import Functions as fn

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
similarity_function = fn.cosine_similarity
test_size = 0.20


#airline_tweets = AirlineTweets(n_trees, n_pairs, similarity_function, test_size)
#airline_tweets.run_bag_of_words()
#airline_tweets.run_tf_idf()

#automotive = Automotive(n_trees, n_pairs, similarity_function, test_size)
#automotive.run_bag_of_words()
#automotive.run_tf_idf()

#blog_authorship_corpus = BlogAuthorshipCorpus(n_trees, n_pairs, similarity_function, test_size)
#blog_authorship_corpus.run_bag_of_words()
#blog_authorship_corpus.run_tf_idf()

#clothing_fit = ClothingFit(n_trees, n_pairs, similarity_function, test_size)
#clothing_fit.run_bag_of_words()
#clothing_fit.run_tf_idf()

#company_reviews = CompanyReviews(n_trees, n_pairs, similarity_function, test_size)
#company_reviews.run_bag_of_words()
#company_reviews.run_tf_idf()

#ecommerce_reviews = EcommerceReviews(n_trees, n_pairs, similarity_function, test_size)
#ecommerce_reviews.run_bag_of_words()
#ecommerce_reviews.run_tf_idf()

#imdb_movies_genre = ImdbMoviesGenre(n_trees, n_pairs, similarity_function, test_size)
#imdb_movies_genre.run_bag_of_words()
#imdb_movies_genre.run_tf_idf()

#indian_politics_news = IndianPoliticsNews(n_trees, n_pairs, similarity_function, test_size)
#indian_politics_news.run_bag_of_words()
#indian_politics_news.run_tf_idf()

#jeopardy_questions = JeopardyQuestions(n_trees, n_pairs, similarity_function, test_size)
#jeopardy_questions.run_bag_of_words()
#jeopardy_questions.run_tf_idf()

#movie_reviews = MovieReviews(n_trees, n_pairs, similarity_function, test_size)
#movie_reviews.run_bag_of_words()
#movie_reviews.run_tf_idf()

#sms_spam = SmsSpam(n_trees, n_pairs, similarity_function, test_size)
#sms_spam.run_bag_of_words()
#sms_spam.run_tf_idf()

#social_recommendation = SocialRecommendation(n_trees, n_pairs, similarity_function, test_size)
#social_recommendation.run_bag_of_words()
#social_recommendation.run_tf_idf()