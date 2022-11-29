from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
# NLTK VADER for sentiment analysis
import nltk
import matplotlib.pyplot as plt
plt.style.use('ggplot')
nltk.downloader.download('vader_lexicon')


class FinViz:
    def __init__(self, stock_ticker):
        self.stock_ticker = stock_ticker
        self.finviz_url = 'https://finviz.com/quote.ashx?t='
        self.plot()

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
        # Instantiate the sentiment intensity analyzer
        vader = SentimentIntensityAnalyzer()

        # Iterate through the headlines and get the polarity scores using vader
        scores = parsed_news_df['headline'].apply(
            vader.polarity_scores).tolist()

        # Convert the 'scores' list of dicts into a DataFrame
        scores_df = pd.DataFrame(scores)

        # Join the DataFrames of the news and the list of dicts
        parsed_and_scored_news = parsed_news_df.join(
            scores_df, rsuffix='_right')
        parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')
        parsed_and_scored_news = parsed_and_scored_news.drop(
            columns=['date', 'time'])
        parsed_and_scored_news = parsed_and_scored_news.rename(
            columns={"compound": "sentiment_score"})

        return parsed_and_scored_news

    def get_stock_news_sentiments(self):
        parsed_news_df = self.get_news_df()
        parsed_and_scored_news = self.score_news(parsed_news_df)
        return parsed_and_scored_news

    def plot_daily_sentiment(self, parsed_and_scored_news, ticker):

        # Group by date and ticker columns from scored_news and calculate the mean
        mean_scores = parsed_and_scored_news.resample('D').mean()
        fig = plt.figure(figsize=(12, 10))
        sns.barplot(data=mean_scores, x=mean_scores.index,
                    y=mean_scores['sentiment_score'], hue='sentiment_score').set(title=f"{self.stock_ticker} Hourly Sentiment Scores")
        fig.autofmt_xdate()
        plt.show()

    def plot(self):
        parsed_and_scored_news = self.get_stock_news_sentiments()
        ticker = self.stock_ticker
        self.plot_daily_sentiment(parsed_and_scored_news, ticker)


if __name__ == "__main__":
    FinViz(stock_ticker='GOOGL')
