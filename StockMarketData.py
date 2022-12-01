import csv
import json
import logging
import os
import sys
import string
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import requests
import matplotlib.image as mpimg
import yfinance as yf
from pandas_datareader import data as web
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from matplotlib import style
import tweepy
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS


class StockMarketDatapipeline:
    """StockMarketDatapipeline Class Pass Proper Stock Ticker as Parameter in order to retrieve data"""

    def __init__(self, stock_ticker: str):
        """StockMarketDatapipeline Constructor"""
        try:
            nltk.download("vader_lexicon")
            style.use("ggplot")
            load_dotenv(find_dotenv())
            self.path = self.get_current_dir()
            self.settings = self.load_settings()
            self.logger = self.app_logger(self.settings["app_name"])
            self.stock_ticker = stock_ticker
            self.bearer_token = os.environ["bearer_token"]
            self.api_key = os.environ["api_key"]
            self.api_key_secrets = os.environ["api_key_secret"]
            self.access_token = os.environ["access_token"]
            self.access_token_secrets = os.environ["access_token_secret"]
            self.logger.info(
                f"Initialize StockMarketDataPipeline for: {stock_ticker}")
        except Exception as e:
            self.logger.error(f"Cannot initialize StockMarketDataPipeline")

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
    def app_logger(name: str) -> logging:
        try:
            path = os.path.dirname(os.path.realpath(__file__))
            log_dir = os.path.join(path, "logs")
            log_file = os.path.join(log_dir, f"{name}.log")
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)

            file_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            console_formatter = logging.Formatter(
                "%(levelname)s -- %(message)s")

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(console_formatter)

            logger = logging.getLogger(name)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.DEBUG)
            return logger
        except OSError:
            raise RuntimeError("Unable to Load App Logger")

    @staticmethod
    def get_current_dir():
        """get current directory"""
        return os.path.dirname(os.path.realpath(__file__))

    def get_stock_data_from_ticker(self) -> pd.DataFrame:
        """Retrieve Data for stock ticker"""
        try:
            self.logger.info(f"Fetching data for {self.stock_ticker}")
            data = yf.download(self.stock_ticker)["Close"]
            data = pd.DataFrame(data, columns=["Close"])
            data.interpolate(
                method="linear", limit_direction="backward", inplace=True)
            data.reset_index(inplace=True)
            cols = {"Date": "Date", "Close": "Price"}
            data.rename(columns=cols, inplace=True)
            self.logger.info("Retrived Stock Ticker Data Successfully")
            return data
        except Exception as e:
            self.logger.error(f"Cannot Get Data from Stock Ticker: {e}")

    def get_realtime_stock_price(self) -> BeautifulSoup:
        """sending get request and retrieving html table using requests and beautiful soup,
        finding price class using beautiful soup and passing html address for price
        """
        self.logger.info(
            f"Getting Realtime Stock Price for {self.stock_ticker}")
        try:
            url = f"{self.settings['yfinance_url']}/{self.stock_ticker}"
            page = requests.get(url)
            soup = BeautifulSoup(page.text, "lxml")
            price = soup.find(
                self.settings["html_address"], class_=self.settings["html_class"]
            ).text
            self.logger.info(
                f"Retrived Realtime Price Successfully {self.stock_ticker}: {price}"
            )
            return float(price)
        except Exception as e:
            self.logger.error(f"Cannot Get Realtime Stock Price for: {e}")

    def plot_stock_ticker_data(self) -> None:
        """plotting line chart for retrieved stock ticker data"""
        try:
            self.logger.info(f"Plotting Stock Data for {self.stock_ticker}")
            data = self.get_stock_data_from_ticker()
            plt.figure(figsize=(10, 7))
            plt.title(
                f"Price History Of {self.stock_ticker}",
                fontsize=15,
                color="black",
                fontweight="bold",
            )
            plt.plot(data["Date"], data["Price"], color="#FE2E2E")
            plt.ylabel("Price in USD", fontsize=15, fontweight="bold")
            plt.xlabel("Date", fontsize=15, fontweight="bold")
            plt.show()
            self.logger.info("Stock Data Plot Successfully")
        except Exception as e:
            self.logger.error(f"{e}: Cannot Plot Data")

    def download_market_info(self) -> None:
        """It let's you download market info in csv file"""
        self.logger.info(f"Downloading Market info for {self.stock_ticker}")
        try:
            sec = yf.Ticker(self.stock_ticker)

            data = sec.history()
            # print(data.head())

            my_max = data["Close"].idxmax()
            my_min = data["Close"].idxmin()
            list_data = [
                data,
                my_max,
                my_min,
                sec.info,
                sec.isin,
                sec.major_holders,
                sec.institutional_holders,
                sec.dividends,
                sec.splits,
                sec.actions,
                sec.calendar,
                sec.recommendations,
                sec.quarterly_earnings,
                sec.earnings,
                sec.financials,
                sec.quarterly_financials,
                sec.balance_sheet,
                sec.quarterly_balance_sheet,
                sec.cashflow,
                sec.quarterly_cashflow,
                sec.sustainability,
                sec.options,
            ]
            with open(f"{(self.stock_ticker).lower()}_market_info.csv", "w") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerows(zip(list_data))
            self.logger.info("Market Info Download Successfully")
        except Exception as e:
            self.logger.error(f"Cannot Download Market Data: {e}")

    def download_stock_ticker_data(self) -> None:
        """It let's you download stock ticker data in Excel File"""
        try:
            self.logger.info(
                f"Downloading Stock Ticker Data {self.stock_ticker}")
            data = self.get_stock_data_from_ticker()
            data["Date"] = pd.to_datetime(data["Date"]).dt.date
            writer = pd.ExcelWriter(
                f"{(self.stock_ticker).lower()}_stock_data.xlsx")
            data.to_excel(writer, sheet_name="stock_data",
                          index=False, na_rep="NaN")
            writer.sheets["stock_data"].set_column(0, 1, 15)
            writer.save()
            self.logger.info(f"Stock Data Downloaded Successfully")
        except Exception as e:
            self.logger.error(f"Cannot Download stock ticker data: {e}")

    @classmethod
    def __get_multiple_stock_data(cls, stock_ticker_list: list):
        df = pd.DataFrame()
        for stock in stock_ticker_list:
            df[stock] = web.DataReader(stock, data_source="yahoo")["Adj Close"]
        df.interpolate(
            method="index", limit_direction="backward", inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Date"}, inplace=True)
        print(df.head())
        return df

    @classmethod
    def plot_multiple_stock_charts(cls, stock_ticker_list):
        data = cls.__get_multiple_stock_data(stock_ticker_list)
        data.plot(subplots=True, figsize=(31, 14))
        plt.legend(loc="best")
        plt.show()

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
            self.__class__.client_auth(self.bearer_token).search_recent_tweets,
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
        print(tweet_list_df.head())
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
        print(tweet_list_df.head())
        return tweet_list_df

    def __seperate_df(self):
        tweet_list_df = self.get_tweets_sentiments()
        tweet_list_df_negative = tweet_list_df[tweet_list_df["sentiment"] == "negative"]
        tweet_list_df_positive = tweet_list_df[tweet_list_df["sentiment"] == "positive"]
        tweet_list_df_neutral = tweet_list_df[tweet_list_df["sentiment"] == "neutral"]

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
        print(data.head())
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
        print("Word Cloud Saved Successfully")
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
            pd.DataFrame(tweet_list_df.groupby("sentiment").text_len.mean()), 2
        )
        word_count = round(
            pd.DataFrame(tweet_list_df.groupby(
                "sentiment").text_word_count.mean()), 2
        )
        print(text_len.head())
        print((word_count.head()))

        return text_len, word_count

    def get_current_stock_open_price(self) -> BeautifulSoup:
        """sending get request and retrieving html table using requests and beautiful soup,
        finding price class using beautiful soup and passing html address for open price
        """
        self.logger.info(f"Getting Current Open Price for {self.stock_ticker}")
        try:
            url = f"{self.settings['yfinance_url']}/{self.stock_ticker}"
            page = requests.get(url)
            soup = BeautifulSoup(page.text, "lxml")
            price = soup.find(
                self.settings["open_address"], class_=self.settings["open_class"]
            ).text
            self.logger.info(
                f"Retrived Open Price Successfully {self.stock_ticker}: {price}"
            )
            return price
        except Exception as e:
            self.logger.error(f"Cannot Get Open Price for: {e}")


if __name__ == "__main__":
    StockMarketDatapipeline("AAPL").plot_tweet_sentiment_donut_chart()
