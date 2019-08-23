
from DataStatistics import Statistics
from datasets.DataType import Type
from datasets.utils import Dataset

from datasets.airline_tweets import AirlineTweetsDataset
from datasets.automotive import AutomotiveDataset
from datasets.blog_authorship_corpus import BlogAuthorshipCorpusDataset
from datasets.clothing_fit_data import ClothingFitDataset
from datasets.company_review_sentence import CompanyReviewSentenceDataset
from datasets.ecommerce_reviews import EcommerceReviewsDataset
from datasets.imdb_movie_genre import IMDBMovieGenreDataset
from datasets.indian_politics_news_2018 import IndianPoliticsNews2018Dataset
from datasets.jeopardy_questions import JeopardyQuestionsDataset
from datasets.movie_reviews import MovieReviewsDataset
from datasets.sms_spam_collection import SmsSpamCollectionDataset
from datasets.social_recommendation_data import SocialRecommendationDataset


# datasets
ds_airline_tweets = AirlineTweetsDataset.load_data(data_type=Type.text, clear_data=True)
ds_automotive = AutomotiveDataset.load_data(data_type=Type.text, clear_data=True)
ds_blog_authorship_corpus = BlogAuthorshipCorpusDataset.load_data(data_type=Type.text, clear_data=True)
ds_clothing_fit = ClothingFitDataset.load_data(data_type=Type.text, clear_data=True)
ds_comapny_review_sentences = CompanyReviewSentenceDataset.load_data(data_type=Type.text, clear_data=True)
ds_ecommerce_reviews = EcommerceReviewsDataset.load_data(data_type=Type.text, clear_data=True)
ds_imdb_movie_genre = IMDBMovieGenreDataset.load_data(data_type=Type.text, clear_data=True)
ds_indian_politics_news = IndianPoliticsNews2018Dataset.load_data(data_type=Type.text, clear_data=True)
ds_jeopardy_questions = JeopardyQuestionsDataset.load_data(data_type=Type.text, clear_data=True)
ds_movie_reviews = MovieReviewsDataset.load_data(data_type=Type.text, clear_data=True)
ds_sms_spam = SmsSpamCollectionDataset.load_data(data_type=Type.text, clear_data=True)
ds_social_recommendation = SocialRecommendationDataset.load_data(data_type=Type.text, clear_data=True)

# statistics
statistics = Statistics()

statistics.add_dataset(ds_airline_tweets)
statistics.add_dataset(ds_automotive)
statistics.add_dataset(ds_blog_authorship_corpus)
statistics.add_dataset(ds_clothing_fit)
statistics.add_dataset(ds_comapny_review_sentences)
statistics.add_dataset(ds_ecommerce_reviews)
statistics.add_dataset(ds_imdb_movie_genre)
statistics.add_dataset(ds_indian_politics_news)
statistics.add_dataset(ds_jeopardy_questions)
statistics.add_dataset(ds_movie_reviews)
statistics.add_dataset(ds_sms_spam)
statistics.add_dataset(ds_social_recommendation)

statistics.apply_number_of_rows()
statistics.apply_longest_sequence()
statistics.apply_average_sequence()
statistics.apply_proportion_of_classes()

statistics.display_number_of_rows()
statistics.display_longest_sequence()
statistics.display_average_sequence()
statistics.display_proportion_of_classes()