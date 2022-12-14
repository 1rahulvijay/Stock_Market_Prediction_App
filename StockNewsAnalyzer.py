import os
import string
import matplotlib.pyplot as plt
import pandas as pd
import requests
import matplotlib.image as mpimg
from dotenv import find_dotenv, load_dotenv
from matplotlib import style
import tweepy
from StockDataAnalyzer import StockDatapipeline
import re
from textblob import TextBlob
import nltk
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import date, timedelta, datetime
import warnings
import matplotlib as mpl
from warnings import simplefilter
style.use('dark_background')
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')
COLOR = 'White'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR



class StockTweets:
    def __init__(self, stock_ticker):
        nltk.download("vader_lexicon")
        self.stock_ticker = stock_ticker
        load_dotenv(find_dotenv())
        self.path = StockDatapipeline.get_current_dir()
        self.settings = StockDatapipeline.load_settings(self.path)
        self.logger = StockDatapipeline.app_logger(self.settings["app_name"])
        self.bearer_token = os.environ["bearer_token"]
        self.api_key = os.environ["api_key"]
        self.api_key_secrets = os.environ["api_key_secret"]
        self.access_token = os.environ["access_token"]
        self.access_token_secrets = os.environ["access_token_secret"]

    @classmethod
    def twitter_auth(cls, access_token, access_token_secret, api_key_secret, api_key):
        try:
            auth = tweepy.OAuth1UserHandler(
                access_token, access_token_secret, api_key_secret, api_key
            )
            api = tweepy.API(auth)
            return api
        except Exception as e:
            raise (f"Cannot Authenticate you: {e}")

    @classmethod
    def client_auth(cls, bearer_token):
        try:
            return tweepy.Client(bearer_token)
        except Exception as e:
            raise (f"Cannot Authenticate Client: {e}")

    @staticmethod
    def remove_puntuations(text):
        punct = string.punctuation
        return text.translate(str.maketrans("", "", punct))

    @staticmethod
    def preprocess_tweet(sen):
        """Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
        in lowercase"""

        sentence = sen.lower()
        sentence = re.sub("RT @\w+: ", " ", sentence)
        sentence = re.sub(
            "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sentence
        )
        sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence

    def get_tweets_df(self):
        try:
            self.logger.info(f"Retrieving Tweets for {self.stock_ticker}")
            tweet_list = []
            query = f"#{self.stock_ticker} -is:retweet lang:en"
            paginator = tweepy.Paginator(
                self.__class__.client_auth(
                    self.bearer_token).search_recent_tweets,
                query=query,
                max_results=100,
                limit=5,
            )
            for tweet in paginator.flatten():
                tweet_list.append(tweet)
            tweet_list_df = pd.DataFrame(tweet_list, columns=["text"])
            return tweet_list_df
        except Exception as e:
            self.logger.error(f"Cannot find any tweet for {self.stock_ticker}")

    def get_cleaned_tweets(self):
        try:
            self.logger.info(f"Cleaning tweets for {self.stock_ticker}")
            cleaned_tweets = []
            tweet_list_df = self.get_tweets_df()
            tweet_list_df["cleaned"] = tweet_list_df["text"].apply(
                self.__class__.preprocess_tweet
            )
            # print(tweet_list_df.head())
            return tweet_list_df
        except Exception as e:
            self.logger.error(
                f"Cannot clean any tweets for {self.stock_ticker}")

    @staticmethod
    def get_polarity(df, column):
        df[["polarity", "subjectivity"]] = df[column].apply(
            lambda Text: pd.Series(TextBlob(Text).sentiment)
        )

        return df

    def get_tweets_polarity(self):
        try:
            tweet_list_df = self.get_cleaned_tweets()
            tweet_list_df = self.get_polarity(tweet_list_df, "cleaned")

            return tweet_list_df
        except Exception as e:
            self.logger.error(f"{e}")

    @staticmethod
    def get_sentiments(df, column):
        try:
            for index, row in df[column].iteritems():
                score = SentimentIntensityAnalyzer().polarity_scores(row)
                neg = score["neg"]
                neu = score["neu"]
                pos = score["pos"]
                comp = score["compound"]
                if comp <= -0.05:
                    df.loc[index, "sentiment"] = "negative"
                elif comp >= 0.05:
                    df.loc[index, "sentiment"] = "positive"
                else:
                    df.loc[index, "sentiment"] = "neutral"
                df.loc[index, "neg"] = neg
                df.loc[index, "neu"] = neu
                df.loc[index, "pos"] = pos
                df.loc[index, "compound"] = comp
            # print(tweet_list_df.head())
            return df
        except OSError:
            raise RuntimeError("Unable to Load df")

    def get_tweets_sentiments(self):
        try:
            tweet_list_df = self.get_tweets_polarity()
            tweet_list_df = self.get_sentiments(tweet_list_df, "cleaned")
            return tweet_list_df
        except Exception as e:
            self.logger.error(f"{e}")

    def __seperate_df(self):
        tweet_list_df = self.get_tweets_sentiments()
        tweet_list_df_negative = tweet_list_df[tweet_list_df["sentiment"] == "negative"]
        tweet_list_df_positive = tweet_list_df[tweet_list_df["sentiment"] == "positive"]
        tweet_list_df_neutral = tweet_list_df[tweet_list_df["sentiment"] == "neutral"]
        return tweet_list_df_negative, tweet_list_df_neutral, tweet_list_df_positive

    @classmethod
    def __count_values_in_column(cls, data, feature):
        total = data.loc[:, feature].value_counts(dropna=False)
        percentage = round(
            data.loc[:, feature].value_counts(
                dropna=False, normalize=True) * 100, 2
        )
        return pd.concat([total, percentage], axis=1, keys=["Total", "Percentage"])

    def get_tweets_sentiments_percentage(self):
        try:
            tweet_list_df = self.get_tweets_sentiments()
            data = self.__class__.__count_values_in_column(
                data=tweet_list_df, feature="sentiment"
            )
            # print(data.head())
            return data
        except Exception as e:
            self.logger.error(f"{e}")

    def plot_tweet_sentiment_donut_chart(self):
        try:
            pichart = self.get_tweets_sentiments_percentage()
            names = pichart.index
            size = pichart['Percentage']
            fig = plt.figure(figsize=(10, 7))
            my_circle = plt.Circle((0, 0), 0.7, color="black")
            plt.pie(size, autopct='%1.0f%%', textprops={'fontsize': 16}, labels=names,
                    colors=['#fab1a0', '#74b9ff', '#ff7675'])
            p = plt.gcf()
            plt.title(label=f'{self.stock_ticker} Tweets Sentiments')
            plt.legend(loc="best")
            p.gca().add_artist(my_circle)
            # plt.show()
            return fig
        except Exception as e:
            self.logger.error(f"{e}")

    def __get_image(self):
        image_temp = os.path.join(
            self.path, self.settings["word_cloud_temp"]
        )
        image_save = os.path.join(
            self.path, self.settings["generated_word_cloud"]
        )
        return image_temp, image_save

    def __create_wordcloud(self, text):
        image_temp, image_save = self.__get_image()
        mask = np.array(Image.open(image_temp))
        stopwords = set(STOPWORDS)
        wc = WordCloud(
            background_color="white",
            mask=mask,
            max_words=100,
            stopwords=stopwords,
            repeat=True,
        )
        wc.generate(str(text))
        wc.to_file(image_save)
        #print("Word Cloud Saved Successfully")
        plt.imshow(mpimg.imread(image_save))

    def plot_word_cloud(self):
        tweet_list_df = self.get_tweets_sentiments()
        self.__create_wordcloud(tweet_list_df["cleaned"].values)

    def get_tweets_word_count_and_len(self):
        tweet_list_df = self.get_tweets_sentiments()
        tweet_list_df["text_len"] = tweet_list_df["cleaned"].astype(
            str).apply(len)
        tweet_list_df["text_word_count"] = tweet_list_df["cleaned"].apply(
            lambda x: len(str(x).split())
        )

        text_len = round(
            pd.DataFrame(tweet_list_df.groupby(
                "sentiment").text_len.mean()), 2
        )
        word_count = round(
            pd.DataFrame(tweet_list_df.groupby(
                "sentiment").text_word_count.mean()), 2
        )
        # print(text_len.head())
        # print((word_count.head()))

        return text_len, word_count


class StockNews(StockTweets):
    def __init__(self, stock_ticker):
        super().__init__(stock_ticker)
        self.finviz_url = 'https://finviz.com/quote.ashx?t='
        #print(date.today(), date.today() - timedelta(days=10))

    def get_news(self):
        try:
            self.logger.info(
                f"Retriving Financial News for {self.stock_ticker}")
            url = self.finviz_url + self.stock_ticker
            req = Request(url=url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
            response = urlopen(req)
            # Read the contents of the file into 'html'
            html = BeautifulSoup(response, "lxml")
            # Find 'news-table' in the Soup and load it into 'news_table'
            news_table = html.find(id='news-table')
            return news_table
        except Exception as e:
            self.logger.error(f"Didn't find news: {e} for {self.stock_ticker}")

    @staticmethod
    def parse_news(news_table):
        parsed_news = []
        for x in news_table.findAll('tr'):
            # read the text from each tr tag into text
            # get text from a only
            text = x.a.get_text()
            # splite text in the td tag into a list
            date_scrape = x.td.text.split()
            # if the length of 'date_scrape' is 1, load 'time' as the only element

            if len(date_scrape) == 1:
                time = date_scrape[0]

            # else load 'date' as the 1st element and 'time' as the second
            else:
                date = date_scrape[0]
                time = date_scrape[1]

            # Append ticker, date, time and headline as a list to the 'parsed_news' list
            parsed_news.append([date, time, text])
            # Set column names
            columns = ['date', 'time', 'headline']
            # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
            parsed_news_df = pd.DataFrame(parsed_news, columns=columns)
            # Create a pandas datetime object from the strings in 'date' and 'time' column
            parsed_news_df['datetime'] = pd.to_datetime(
                parsed_news_df['date'] + ' ' + parsed_news_df['time'])

        return parsed_news_df

    def get_news_df(self):
        self.logger.info(f"parsing news for {self.stock_ticker}")
        try:
            news_table = self.get_news()
            parsed_news_df = self.parse_news(news_table)
            return parsed_news_df
        except Exception as e:
            self.logger.error(f"Didn't find any news for {self.stock_ticker}")

    def get_stock_news_sentiments(self):
        parsed_news_df = self.get_news_df()
        parsed_and_scored_news = super(
            StockNews, self).get_polarity(parsed_news_df, 'headline')

        parsed_and_scored_news = super(StockNews, self).get_sentiments(
            parsed_and_scored_news, 'headline')

        parsed_and_scored_news = parsed_news_df.set_index('datetime')

        parsed_and_scored_news = parsed_and_scored_news.drop(
            columns=['date', 'time'])

        parsed_and_scored_news = parsed_and_scored_news.rename(
            columns={"compound": "sentiment_score"})

        parsed_and_scored_news.sort_index(ascending=False, inplace=True)
        return parsed_and_scored_news

    def __plot_daily_sentiment(self, parsed_and_scored_news, ticker):

        # Group by date and ticker columns from scored_news and calculate the mean
        df = parsed_and_scored_news
        df2 = df.reset_index()
        df2['datetime'] = df2['datetime'].dt.date
        df2.set_index('datetime', inplace=True)
        df2.sort_index(ascending=True, inplace=True)
        color = ['#d63031', '#55efc4', '#74b9ff']
        fig = plt.figure(figsize=(11, 3))
        sns.barplot(data=df2, x=df2.index,
                    y=df2['sentiment_score'], hue=df2['sentiment'], palette=color)
        fig.autofmt_xdate()
        plt.title(label=f"{self.stock_ticker} Daily News Sentiment Scores")
        # plt.show()
        return fig

    def plot_daily_sentiment_barchart(self):
        try:
            self.logger.info(
                f"Plotting Daily Sentiment Bar Chart for {self.stock_ticker}")
            parsed_and_scored_news = self.get_stock_news_sentiments()
            ticker = self.stock_ticker
            return self.__plot_daily_sentiment(parsed_and_scored_news, ticker)
        except Exception as e:
            self.logger.error(f"Couldn't plot any chart {e}")

    def __get_sentiments_with_price(self):
        parsed_and_scored_news = self.get_stock_news_sentiments()
        index = parsed_and_scored_news.index.date
        start = str(index.max())
        end = str(index.min())
        df = yf.download(self.stock_ticker, end, start)
        df2 = parsed_and_scored_news.resample('D').mean()
        news_with_price = df2.join(df['Close']).dropna()
        news_with_price.sort_index(ascending=True, inplace=True)
        return news_with_price

    def plot_sentiments_with_price(self):
        try:
            self.logger.info(
                f"Plotting Sentiments with price chart for {self.stock_ticker}")
            df = self.__get_sentiments_with_price()
            fig = plt.figure(figsize=(7, 3))
            #sns.set(rc={'figure.figsize': (2, 3)})
            ax = sns.lineplot(data=df['Close'], color="#74b9ff",
                              label=f'{self.stock_ticker} Price')
            ax2 = plt.twinx()
            sns.lineplot(data=df["sentiment_score"],
                         color="red", ax=ax2, label=f'{self.stock_ticker} Sentiment Score from News')

            plt.title(
                label=f"{self.stock_ticker} Price Affected by Daily Stock News Sentiments")
            fig.autofmt_xdate()
            # plt.show()
            return fig
        except Exception as e:
            self.logger.error(
                f"Couldn't plot Sentiments with price for {self.stock_ticker}")


if __name__ == "__main__":
    # StockTweets(stock_ticker='MSFT').plot_tweet_sentiment_donut_chart()
    StockNews(stock_ticker='IBM').plot_word_cloud()
