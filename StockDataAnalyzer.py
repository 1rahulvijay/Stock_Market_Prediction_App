import csv
import json
import logging
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import requests
import matplotlib.image as mpimg
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
import numpy as np
import seaborn
from scipy import signal
from dateutil.parser import parse
import matplotlib as mpl
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.pyplot import style
from warnings import simplefilter
import warnings
import matplotlib as mpl
from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left': False, 'axes.titlepad': 10})
# plt.rcParams.update({'font.size': 4})
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
style.use('dark_background')
warnings.filterwarnings('ignore')
COLOR = 'White'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR


class StockDatapipeline:
    """StockMarketDatapipeline Class Pass Proper Stock Ticker as Parameter in order to retrieve data"""

    def __init__(self, stock_ticker: str):
        """StockMarketDatapipeline Constructor"""
        try:
            load_dotenv(find_dotenv())
            self.path = self.get_current_dir()
            self.settings = self.load_settings(self.path)
            self.logger = self.app_logger(self.settings["app_name"])
            self.stock_ticker = stock_ticker
            self.data = self.get_stock_data_from_ticker(stock_ticker)
            # self.logger.info(
            #     f"Initialize StockMarketDataPipeline for: {stock_ticker}")
        except Exception as e:
            self.logger.error(f"Cannot initialize StockMarketDataPipeline")

    @staticmethod
    def load_settings(path) -> json:
        try:
            settings = dict()
            json_path = os.path.join(path, "data\data.json")
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

    @staticmethod
    def get_stock_data_from_ticker(stock_ticker):
        start = "2019-01-01"
        end = datetime.today().strftime('%Y-%m-%d')
        data = yf.download(stock_ticker, start, end)
        data.interpolate(
            method="linear", limit_direction="backward", inplace=True)
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'Date'}, inplace=True)
        return data

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
            data = self.data
            fig = plt.figure(figsize=(6, 3))
            plt.title(
                f"Price History Of {self.stock_ticker}",
                fontsize=15,
                color="black",
                fontweight="bold",
            )
            plt.plot(data["Date"], data["Close"], color="#FE2E2E")
            plt.ylabel("Price in USD", fontsize=15, fontweight="bold")
            plt.xlabel("Date", fontsize=15, fontweight="bold")
            return fig
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
            with open(f"{(self.stock_ticker).upper()}_Market_Info.csv", "w") as f:
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
            data = self.data
            data["Date"] = pd.to_datetime(data["Date"]).dt.date
            writer = pd.ExcelWriter(
                f"{(self.stock_ticker).lower()}_stock_data.xlsx")
            data.to_excel(writer, sheet_name="stock_data",
                          index=False, na_rep="NaN")
            writer.sheets["stock_data"].set_column(0, 5, 15)
            writer.save()
            self.logger.info(f"Stock Data Downloaded Successfully")
        except Exception as e:
            self.logger.error(f"Cannot Download stock ticker data: {e}")

    def get_current_stock_open_price(self) -> BeautifulSoup:
        """Reading Open Price using Read HTML
        """
        self.logger.info(f"Getting Current Open Price for {self.stock_ticker}")
        try:
            url = f"{self.settings['yfinance_url']}/{self.stock_ticker}"
            df = pd.read_html(url)
            df = df[0]
            open = df[1].iloc[1]
            return float(open)
        except Exception as e:
            self.logger.error(f"Cannot Get Open Price for: {e}")

    def get_stock_volume(self) -> BeautifulSoup:
        """Reading Volume from Read HTML
        """
        self.logger.info(f"Getting Stock Volume for {self.stock_ticker}")
        try:
            url = f"{self.settings['yfinance_url']}/{self.stock_ticker}"
            df = pd.read_html(url)
            df = df[0]
            volume = df[1].iloc[6]
            return float(volume)
        except Exception as e:
            self.logger.error(f"Cannot Get Stock Volume: {e}")

    def __plot_trend(self, df, x, y, title=f"", xlabel='Date', ylabel='Stock price', dpi=200):
        fig = plt.figure(figsize=(10, 2), dpi=dpi)
        plt.plot(x, y, color='tab:Red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.legend([self.stock_ticker])
        plt.ion()
        return fig

    def plot_trend(self):
        df = self.data
        return self.__plot_trend(df, x=df['Date'], y=df['Close'],
                                 title=f'Trend and Seasonality {self.stock_ticker}')

    def plot_two_side_view(self):
        df = self.data
        x = df['Date'].values
        y1 = df['Close'].values
        fig, ax = plt.subplots(1, 1, figsize=(10.5, 10), dpi=120)
        plt.fill_between(x, y1=y1, y2=-y1, alpha=0.5,
                         linewidth=2, color='#FF4B4B')
        #plt.ylim(-800, 800)
        plt.title(f'{self.stock_ticker} (Two Side View)', fontsize=16)
        plt.hlines(y=0, xmin=np.min(df['Date']),
                   xmax=np.max(df['Date']), linewidth=.5)
        fig.autofmt_xdate()
        return fig

    def plot_multiplicative_decompose(self):
        # Multiplicative Decomposition
        df = self.data
        multiplicative_decomposition = seasonal_decompose(
            df['Close'], model='multiplicative', period=30)

        # Plot
        fig = plt.figure()
        plt.rcParams.update({'figure.figsize': (10, 7)})
        multiplicative_decomposition.plot().suptitle(
            'Multiplicative Decomposition', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig

    def plot_additive_decomposition(self):
        df = self.data
        # Additive Decomposition
        additive_decomposition = seasonal_decompose(
            df['Close'], model='additive', period=30)

        fig = plt.figure()
        plt.rcParams.update({'figure.figsize': (10, 7)})
        additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig

    def plot_detrend(self):
        df = self.data
        detrended = signal.detrend(df['Close'].values)
        fig = plt.figure(figsize=(7, 5))
        plt.plot(detrended)
        plt.title(
            f'{self.stock_ticker} Price detrended by subtracting the least squares fit', fontsize=16)
        return fig

    def plot_detrend_seasonal_decompose(self):
        df = self.data
        result_mul = seasonal_decompose(
            df['Close'], model='multiplicative', period=30)
        detrended = df['Close'].values - result_mul.trend
        fig = plt.figure(figsize=(7, 5))
        plt.plot(detrended)
        plt.title(
            f'{self.stock_ticker} Price detrended by subtracting the trend component', fontsize=16)
        return fig

    def plot_price_deseasonlized(self):
        df = self.data
        result_mul = seasonal_decompose(
            df['Close'], model='multiplicative', period=30)

        # Deseasonalize
        deseasonalized = df['Close'].values / result_mul.seasonal

        # Plot
        fig = plt.figure(figsize=(10, 7))
        plt.plot(deseasonalized)
        plt.title(f'{self.stock_ticker} Price Deseasonalized', fontsize=16)
        return fig

    def plot_autocorrelation(self):
        df = self.data
        fig = plt.rcParams.update(
            {'figure.figsize': (10, 7), 'figure.dpi': 120})
        autocorrelation_plot(df['Close'].tolist())
        plt.title(
            f'{self.stock_ticker} Autocorrelation', fontsize=16)
        return fig

    def plot_acf_pacf(self):
        df = self.data
        fig, axes = plt.subplots(1, 2, figsize=(10, 7), dpi=100)
        plot_acf(df['Close'].tolist(), lags=50, ax=axes[0],
                 title=f'{self.stock_ticker} Autocorrelation')
        plot_pacf(df['Close'].tolist(), lags=50, ax=axes[1],
                  title=f'{self.stock_ticker} Partial Autocorrelation')
        return fig

    def plot_lag(self):
        df = self.data
        fig, axes = plt.subplots(1, 4, figsize=(
            10, 7), sharex=True, sharey=True, dpi=100)
        for i, ax in enumerate(axes.flatten()[:4]):
            lag_plot(df['Close'], lag=i+1, ax=ax, c='firebrick')
            ax.set_title('Lag ' + str(i+1))

        fig.suptitle(f'Lag Plots of {self.stock_ticker}', y=1.05)
        return fig

    def __get_ema(self):
        df = self.data
        df['SMA_50'] = df['Close'].rolling(50).mean().shift()
        df['SMA_100'] = df['Close'].rolling(100).mean().shift()
        df['SMA_200'] = df['Close'].rolling(200).mean().shift()
        return df

    @staticmethod
    def __plot_sma(df, stock_ticker, sma, days):
        df.set_index(df['Date'], inplace=True)
        fig = plt.figure(figsize=(7, 2))
        plt.plot(df['Close'], 'k-', label='Original', color='#74b9ff')
        plt.plot(df[f'{sma}'], 'r-', label='Running average')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.grid(linestyle=':')
        plt.fill_between(df.index, 0, df[f'{sma}'], color='tab:Red', alpha=0.1)
        plt.legend(loc='lower left')
        plt.title(f'{stock_ticker} {days} Days Simple Moving Average')
        # plt.show()
        return fig

    def plot_100_days_sma(self):
        df = self.__get_ema()
        return self.__plot_sma(df=df, stock_ticker=self.stock_ticker,
                               sma='SMA_100', days=100)

    def plot_200_days_sma(self):
        df = self.__get_ema()
        return self.__plot_sma(df=df, stock_ticker=self.stock_ticker,
                               sma='SMA_200', days=200)

    def plot_50_days_sma(self):
        df = self.__get_ema()
        return self.__plot_sma(df=df, stock_ticker=self.stock_ticker,
                               sma='SMA_50', days=50)

    @staticmethod
    def __plot_combined_moving_average(df, ma1, ma2, title):
        ma_1 = pd.Series.rolling(df.Close, window=ma1).mean()
        ma_2 = pd.Series.rolling(df.Close, window=ma2).mean()

        xdate = [x for x in df.Date]

        fig = plt.figure(figsize=(6, 3))

        plt.plot(xdate, df.Close, lw=1, color="black", label="Price")
        plt.plot(xdate, ma_1, lw=3, linestyle="dotted",
                 label="Moving Average {} days".format(ma1))
        plt.plot(xdate, ma_2, lw=3, linestyle="dotted",
                 label="Moving Average {} days".format(ma2))
        plt.legend(loc='best')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(title)
        return fig

    def plot_week_and_month_moving_average(self):
        df = self.data
        return self.__plot_combined_moving_average(
            df=df, ma1=7, ma2=30, title=f"{self.stock_ticker} 7 Days and 30 Days Moving Average")

    @staticmethod
    def __relative_strength_idx(df, n):
        close = df['Close']
        delta = close.diff()
        delta = delta[1:]
        pricesUp = delta.copy()
        pricesDown = delta.copy()
        pricesUp[pricesUp < 0] = 0
        pricesDown[pricesDown > 0] = 0
        rollUp = pricesUp.rolling(n).mean()
        rollDown = pricesDown.abs().rolling(n).mean()
        rs = rollUp / rollDown
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def plot_rsi(self):
        df = self.data
        df['RSI'] = self.__relative_strength_idx(df=df, n=14)
        fig = plt.figure(figsize=(7, 2))
        plt.plot(df['Date'], df['RSI'], label="RSI", color='tab:Red')
        plt.ylabel('RSI')
        plt.xlabel('Date')
        plt.grid(linestyle=':')
        plt.legend(loc='upper left')
        plt.title(f'{self.stock_ticker} Relative Strength Index')
        return fig

    @staticmethod
    def __get_macd_and_ema(df):
        EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
        EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
        df['MACD'] = pd.Series(EMA_12 - EMA_26)
        df['MACD_signal'] = pd.Series(
            df.MACD.ewm(span=9, min_periods=9).mean())
        return df

    def plot_macd(self):
        df = self.__get_macd_and_ema(self.data)
        fig = plt.figure(figsize=(7, 2))
        plt.plot(df['Date'], df['MACD'], label='MACD', color='tab:Red')
        plt.plot(df['Date'], df['MACD_signal'], label='MACD Signal', color='#74b9ff')
        plt.xlabel('Date')
        plt.grid(linestyle=':')
        plt.legend(loc='upper left')
        plt.title(f'{self.stock_ticker} MACD and MACD Signal')
        return fig


if __name__ == "__main__":
    StockDatapipeline("AAPL").get_previous_stock_close()
