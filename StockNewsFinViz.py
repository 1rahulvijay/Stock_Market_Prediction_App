from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
from textblob import TextBlob
from datetime import date, timedelta, datetime
# NLTK VADER for sentiment analysis
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta, datetime
import pandas_datareader as web
plt.style.use('ggplot')
nltk.downloader.download('vader_lexicon')


class FinViz:
    def __init__(self, stock_ticker):
        self.stock_ticker = stock_ticker
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
        color = ['#74b9ff','#55efc4', '#d63031']
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
        df = web.get_data_stooq(self.stock_ticker, end, start)
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
    FinViz(stock_ticker='META').plot_sentiments_with_price()