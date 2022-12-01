import csv
import json
import logging
import os
import sys
import string
import matplotlib.pyplot as plt
import pandas as pd
import requests
import matplotlib.image as mpimg
from dotenv import find_dotenv, load_dotenv
from matplotlib import style
import tweepy
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


class StockTweets():
    def __init__(self, stock_ticker):
        self.stock_ticker = stock_ticker
        nltk.download("vader_lexicon")
        style.use("ggplot")
        load_dotenv(find_dotenv())
        self.path = self.get_current_dir()
        self.settings = self.load_settings()
        self.stock_ticker = stock_ticker
        self.bearer_token = os.environ["bearer_token"]
        self.api_key = os.environ["api_key"]
        self.api_key_secrets = os.environ["api_key_secret"]
        self.access_token = os.environ["access_token"]
        self.access_token_secrets = os.environ["access_token_secret"]

    def load_settings(self) -> json:
        try:
            settings = dict()
            json_path = os.path.join(self.path, "data\data.json")
            with open(json_path, "r") as file:
                settings = json.load(file)
                return settings
        except OSError:
            raise RuntimeError("Cannot Load JSON Data File")

    @staticmethod
    def get_current_dir():
        """get current directory"""
        return os.path.dirname(os.path.realpath(__file__))

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

    def get_cleaned_tweets(self):
        cleaned_tweets = []
        tweet_list_df = self.get_tweets_df()
        tweet_list_df["cleaned"] = tweet_list_df["text"].apply(
            self.__class__.preprocess_tweet
        )
        # print(tweet_list_df.head())
        return tweet_list_df

    def get_tweets_polarity(self):
        tweet_list_df = self.get_cleaned_tweets()
        tweet_list_df[["polarity", "subjectivity"]] = tweet_list_df["cleaned"].apply(
            lambda Text: pd.Series(TextBlob(Text).sentiment)
        )

        return tweet_list_df

    def get_tweets_sentiments(self):
        tweet_list_df = self.get_tweets_polarity()
        for index, row in tweet_list_df["cleaned"].iteritems():
            score = SentimentIntensityAnalyzer().polarity_scores(row)
            neg = score["neg"]
            neu = score["neu"]
            pos = score["pos"]
            comp = score["compound"]
            if comp <= -0.05:
                tweet_list_df.loc[index, "sentiment"] = "negative"
            elif comp >= 0.05:
                tweet_list_df.loc[index, "sentiment"] = "positive"
            else:
                tweet_list_df.loc[index, "sentiment"] = "neutral"
            tweet_list_df.loc[index, "neg"] = neg
            tweet_list_df.loc[index, "neu"] = neu
            tweet_list_df.loc[index, "pos"] = pos
            tweet_list_df.loc[index, "compound"] = comp
        # print(tweet_list_df.head())
        return tweet_list_df

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
        tweet_list_df = self.get_tweets_sentiments()
        data = self.__class__.__count_values_in_column(
            data=tweet_list_df, feature="sentiment"
        )
        # print(data.head())
        return data

    def plot_tweet_sentiment_donut_chart(self):
        pichart = self.get_tweets_sentiments_percentage()
        names = pichart.index
        size = pichart['Percentage']
        my_circle = plt.Circle((0, 0), 0.7, color="white")
        plt.pie(size, autopct='%1.0f%%', textprops={'fontsize': 16}, labels=names,
                colors=['#fab1a0', '#74b9ff', '#ff7675'])
        p = plt.gcf()
        plt.legend(loc="best")
        p.gca().add_artist(my_circle)
        plt.show()

    def __get_image(self):
        image_temp = os.path.join(
            self.get_current_dir(), self.settings["word_cloud_temp"]
        )
        image_save = os.path.join(
            self.get_current_dir(), self.settings["generated_word_cloud"]
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
        plt.show()

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
        print(date.today(), date.today() - timedelta(days=10))

    def get_news(self):
        url = self.finviz_url + self.stock_ticker
        req = Request(url=url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
        response = urlopen(req)
        # Read the contents of the file into 'html'
        html = BeautifulSoup(response, "lxml")
        # Find 'news-table' in the Soup and load it into 'news_table'
        news_table = html.find(id='news-table')
        return news_table

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
        news_table = self.get_news()
        parsed_news_df = self.parse_news(news_table)
        return parsed_news_df

    @staticmethod
    def score_news(parsed_news_df):
        parsed_news_df[["polarity", "subjectivity"]] = parsed_news_df["headline"].apply(
            lambda Text: pd.Series(TextBlob(Text).sentiment)
        )

        for index, row in parsed_news_df["headline"].iteritems():
            score = SentimentIntensityAnalyzer().polarity_scores(row)
            neg = score["neg"]
            neu = score["neu"]
            pos = score["pos"]
            comp = score["compound"]
            if comp <= -0.05:
                parsed_news_df.loc[index, "sentiment"] = "negative"
            elif comp >= 0.05:
                parsed_news_df.loc[index, "sentiment"] = "positive"
            else:
                parsed_news_df.loc[index, "sentiment"] = "neutral"
            parsed_news_df.loc[index, "neg"] = neg
            parsed_news_df.loc[index, "neu"] = neu
            parsed_news_df.loc[index, "pos"] = pos
            parsed_news_df.loc[index, "compound"] = comp
        # print(parsed_news_df.head())

        parsed_and_scored_news = parsed_news_df.set_index('datetime')
        # print(parsed_and_scored_news)
        parsed_and_scored_news = parsed_and_scored_news.drop(
            columns=['date', 'time'])
        parsed_and_scored_news = parsed_and_scored_news.rename(
            columns={"compound": "sentiment_score"})
        # print(parsed_and_scored_news)

        return parsed_and_scored_news

    def get_stock_news_sentiments(self):
        parsed_news_df = self.get_news_df()
        parsed_and_scored_news = self.score_news(parsed_news_df)
        # print(parsed_and_scored_news)
        return parsed_and_scored_news

    def plot_daily_sentiment(self, parsed_and_scored_news, ticker):

        # Group by date and ticker columns from scored_news and calculate the mean
        df = parsed_and_scored_news
        df2 = df.reset_index()
        df2['datetime'] = df2['datetime'].dt.date
        df2.set_index('datetime', inplace=True)
        #df2.sort_index(ascending=True, inplace=True)
        # mean_scores = df.resample('D').mean()
        # print(mean_scores)
        # mean_scores.join(df2['sentiment'], how='left')
        # print(mean_scores)
        # print(mean_scores)
        color = ['#74b9ff', '#55efc4', '#d63031']
        fig = plt.figure(figsize=(12, 10))
        sns.barplot(data=df2, x=df2.index,
                    y=df2['sentiment_score'], hue=df2['sentiment'], palette=color)
        fig.autofmt_xdate()
        plt.title(label=f"{self.stock_ticker} Daily Sentiment Scores")

        plt.show()

    def plot_daily_sentiment_barchart(self):
        parsed_and_scored_news = self.get_stock_news_sentiments()
        ticker = self.stock_ticker
        self.plot_daily_sentiment(parsed_and_scored_news, ticker)
        return parsed_and_scored_news

    def get_sentiments_with_price(self):
        parsed_and_scored_news = self.get_stock_news_sentiments()
        index = parsed_and_scored_news.index.date
        start = str(index.max())
        end = str(index.min())
        df = yf.download(self.stock_ticker, end, start)
        df2 = parsed_and_scored_news.resample('D').mean()
        news_with_price = df2.join(df['Close']).dropna()
        return news_with_price

    def plot_sentiments_with_price(self):
        df = self.get_sentiments_with_price()
        sns.set(rc={'figure.figsize': (13.0, 8.0)})
        ax = sns.lineplot(data=df['Close'], color="green",
                          label=f'{self.stock_ticker} Price')
        ax2 = plt.twinx()
        sns.lineplot(data=df["sentiment_score"],
                     color="red", ax=ax2, label=f'{self.stock_ticker} Sentiment Score from News')

        plt.title(
            label=f"{self.stock_ticker} Price with Stock News Sentiments")

        plt.show()


if __name__ == "__main__":
    StockNews(stock_ticker='GOOGL').plot_sentiments_with_price()
    #StockTweets(stock_ticker='GOOGL').plot_word_cloud()
