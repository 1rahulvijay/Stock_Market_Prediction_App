import os
import yfinance as yf
import numpy as np
import pandas as pd
import seaborn
from scipy import signal
from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.pyplot import style
from warnings import simplefilter
import warnings
from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left': False, 'axes.titlepad': 10})
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
style.use('ggplot')
warnings.filterwarnings('ignore')


class TimeSeriesDecompose:
    def __init__(self, stock_ticker):
        self.stock_ticker = stock_ticker
        self.data = self.get_stock_data_from_ticker(self.stock_ticker)

    @staticmethod
    def get_stock_data_from_ticker(stock_ticker):
        data = yf.download(stock_ticker)
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'Date'}, inplace=True)
        return data

    def __plot_trend(self, df, x, y, title=f"", xlabel='Date', ylabel='Stock price', dpi=100):
        plt.figure(figsize=(15, 4), dpi=dpi)
        plt.plot(x, y, color='tab:Red')
        plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
        plt.legend([self.stock_ticker])
        plt.show()

    def plot_trend(self):
        df = self.data
        self.__plot_trend(df, x=df['Date'], y=df['Close'],
                          title=f'Trend and Seasonality for {self.stock_ticker}')

    def plot_two_side_view(self):
        df = self.data
        x = df['Date'].values
        y1 = df['Close'].values
        fig, ax = plt.subplots(1, 1, figsize=(16, 5), dpi=120)
        plt.fill_between(x, y1=y1, y2=-y1, alpha=0.5,
                         linewidth=2, color='seagreen')
        #plt.ylim(-800, 800)
        plt.title(f'{self.stock_ticker} (Two Side View)', fontsize=16)
        plt.hlines(y=0, xmin=np.min(df['Date']),
                   xmax=np.max(df['Date']), linewidth=.5)
        plt.show()

    def plot_multiplicative_decompose(self):
        # Multiplicative Decomposition
        df = self.data
        multiplicative_decomposition = seasonal_decompose(
            df['Close'], model='multiplicative', period=30)

        # Plot
        plt.rcParams.update({'figure.figsize': (16, 12)})
        multiplicative_decomposition.plot().suptitle(
            'Multiplicative Decomposition', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()

    def plot_additive_decomposition(self):
        df = self.data
        # Additive Decomposition
        additive_decomposition = seasonal_decompose(
            df['Close'], model='additive', period=30)

        plt.rcParams.update({'figure.figsize': (16, 12)})
        additive_decomposition.plot().suptitle('Additive Decomposition', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show()

    def plot_detrend(self):
        df = self.data
        detrended = signal.detrend(df['Close'].values)
        plt.plot(detrended)
        plt.title(
            f'{self.stock_ticker} Price detrended by subtracting the least squares fit', fontsize=16)
        plt.show()

    def plot_detrend_seasonal_decompose(self):
        df = self.data
        result_mul = seasonal_decompose(
            df['Close'], model='multiplicative', period=30)
        detrended = df['Close'].values - result_mul.trend
        plt.plot(detrended)
        plt.title(
            f'{self.stock_ticker} Price detrended by subtracting the trend component', fontsize=16)
        plt.show()

    def plot_price_deseasonlized(self):
        df = self.data
        result_mul = seasonal_decompose(
            df['Close'], model='multiplicative', period=30)

        # Deseasonalize
        deseasonalized = df['Close'].values / result_mul.seasonal

        # Plot
        plt.plot(deseasonalized)
        plt.title(f'{self.stock_ticker} Price Deseasonalized', fontsize=16)
        plt.show()

    def plot_autocorrelation(self):
        df = self.data
        plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 120})
        autocorrelation_plot(df['Close'].tolist())
        plt.title(
            f'{self.stock_ticker} Autocorrelation', fontsize=16)
        plt.show()

    def plot_acf_pacf(self):
        df = self.data
        fig, axes = plt.subplots(1, 2, figsize=(16, 3), dpi=100)
        plot_acf(df['Close'].tolist(), lags=50, ax=axes[0],
                 title=f'{self.stock_ticker} Autocorrelation')
        plot_pacf(df['Close'].tolist(), lags=50, ax=axes[1],
                  title=f'{self.stock_ticker} Partial Autocorrelation')
        plt.show()

    def plot_lag(self):
        df = self.data
        fig, axes = plt.subplots(1, 4, figsize=(
            10, 3), sharex=True, sharey=True, dpi=100)
        for i, ax in enumerate(axes.flatten()[:4]):
            lag_plot(df['Close'], lag=i+1, ax=ax, c='firebrick')
            ax.set_title('Lag ' + str(i+1))

        fig.suptitle(f'Lag Plots of {self.stock_ticker}', y=1.05)
        plt.show()

    def __get_ema(self):
        df = self.data
        df['SMA_50'] = df['Close'].rolling(50).mean().shift()
        df['SMA_100'] = df['Close'].rolling(100).mean().shift()
        df['SMA_200'] = df['Close'].rolling(200).mean().shift()
        return df

    @staticmethod
    def __plot_sma(df, stock_ticker, sma, days):
        plt.figure(figsize=(10, 5))
        plt.plot(df['Close'], 'k-', label='Original')
        plt.plot(df[f'{sma}'], 'r-', label='Running average')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.grid(linestyle=':')
        plt.fill_between(df.index, 0, df[f'{sma}'], color='r', alpha=0.1)
        plt.legend(loc='upper left')
        plt.title(f'{stock_ticker} {days} Days Simple Moving Average')
        plt.show()

    def plot_100_days_sma(self):
        df = self.__get_ema()
        self.__plot_sma(df=df, stock_ticker=self.stock_ticker,
                        sma='SMA_100', days=100)

    def plot_200_days_sma(self):
        df = self.__get_ema()
        self.__plot_sma(df=df, stock_ticker=self.stock_ticker,
                        sma='SMA_200', days=200)

    def plot_50_days_sma(self):
        df = self.__get_ema()
        self.__plot_sma(df=df, stock_ticker=self.stock_ticker,
                        sma='SMA_50', days=50)

    @staticmethod
    def __plot_combined_moving_average(df, ma1, ma2, title):
        ma_1 = pd.Series.rolling(df.Close, window=ma1).mean()
        ma_2 = pd.Series.rolling(df.Close, window=ma2).mean()

        xdate = [x for x in df.Date]

        plt.figure(figsize=(15, 5))
        plt.style.use('ggplot')

        plt.plot(xdate, df.Close, lw=1, color="black", label="Price")
        plt.plot(xdate, ma_1, lw=3, linestyle="dotted",
                 label="Moving Average {} days".format(ma1))
        plt.plot(xdate, ma_2, lw=3, linestyle="dotted",
                 label="Moving Average {} days".format(ma2))
        plt.legend(loc='best')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(title)
        plt.show()

    def plot_week_and_month_moving_average(self):
        df = self.data
        self.__plot_combined_moving_average(
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
        plt.figure(figsize=(10, 5))
        plt.plot(df['Date'], df['RSI'], label="RSI")
        plt.ylabel('RSI')
        plt.xlabel('Date')
        plt.grid(linestyle=':')
        plt.legend(loc='upper left')
        plt.title(f'{self.stock_ticker} Relative Strength Index')
        plt.show()

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
        plt.figure(figsize=(10, 5))
        plt.plot(df['Date'], df['MACD'], label='MACD')
        plt.plot(df['Date'], df['MACD_signal'], label='MACD Signal')
        plt.xlabel('Date')
        plt.grid(linestyle=':')
        plt.legend(loc='upper left')
        plt.title(f'{self.stock_ticker} MACD and MACD Signal')
        plt.show()


TimeSeriesDecompose(stock_ticker='BTC-USD').plot_multiplicative_decompose()