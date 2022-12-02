import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from warnings import simplefilter
from StockDataAnalyzer import StockDatapipeline
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)


class XGBoostModel:
    def __init__(self, stock_ticker):
        self.test_size = 0.15
        self.valid_size = 0.15
        self.stock_ticker = stock_ticker
        self.data = StockDatapipeline.get_stock_data_from_ticker(stock_ticker)
        self.gamma = 0.01
        self.learning_rate = 0.05
        self.max_depth = 8
        self.n_estimators = 400
        self.random_state = 40

    @staticmethod
    def get_moving_average(df, column):
        df['EMA_7'] = df[column].ewm(7).mean().shift()
        df['EMA_15'] = df[column].ewm(15).mean().shift()
        df['EMA_30'] = df[column].ewm(30).mean().shift()
        df['SMA_50'] = df[column].rolling(50).mean().shift()
        df['SMA_7'] = df[column].rolling(7).mean().shift()
        df['SMA_30'] = df[column].rolling(30).mean().shift()
        return df

    def get_data(self):
        df = self.data
        df_close = df[['Date', 'Close']].copy()
        df_close = df_close.set_index('Date')
        new_df = self.get_moving_average(df_close, 'Close')
        new_df.dropna(inplace=True)
        # print(new_df)
        return new_df

    def split_data(self):
        drop_cols = ['Date']
        df = self.get_data()
        df.reset_index(inplace=True)
        print(df)

        test_split_idx = int(round(df.shape[0] * (1-self.test_size)))
        valid_split_idx = int(round(
            df.shape[0] * (1-(self.valid_size+self.test_size))))

        print(test_split_idx)
        print(valid_split_idx)

        train_df = df.loc[:valid_split_idx].copy()
        valid_df = df.loc[valid_split_idx+1:test_split_idx].copy()
        test_df = df.loc[test_split_idx+1:].copy()

        train_df = train_df.drop(columns=drop_cols)
        valid_df = valid_df.drop(columns=drop_cols)
        test_df = test_df.drop(columns=drop_cols)

        y_train = train_df['Close'].copy()
        X_train = train_df.drop(columns=['Close'])

        y_valid = valid_df['Close'].copy()
        X_valid = valid_df.drop(columns=['Close'])

        y_test = test_df['Close'].copy()
        X_test = test_df.drop(columns=['Close'])

        print(y_train, X_train, y_valid, X_valid, y_test, X_test)
        return y_train, X_train, y_valid, X_valid, y_test, X_test

    def build_and_train_model(self):
        y_train, X_train, y_valid, X_valid, y_test, X_test = self.split_data()
        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        model = xgb.XGBRegressor(gamma=self.gamma, learning_rate=self.learning_rate,
                                 max_depth=self.max_depth, n_estimators=self.n_estimators,
                                 random_state=self.random_state, objective='reg:squarederror')
        xg = model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return xg

    def plot_importance_of_feature(self):
        model = self.build_and_train_model()
        plot_importance(model)
        plt.show()

    @staticmethod
    def y_pred(model, X_test):
        y_pred = model.predict(X_test)
        # print(f'y_true = {np.array(y_test)[:5]}')
        # print(f'y_pred = {y_pred[:5]}')
        return y_pred

    def get_ypredicted_value(self):
        y_train, X_train, y_valid, X_valid, y_test, X_test = self.split_data()
        model = self.build_and_train_model()
        y_pred = self.y_pred(model=model, X_test=X_test)
        print(y_pred)
        print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')
        return y_pred

    def plot_xgboost_prediction(self):
        df = self.get_data()
        df.reset_index(inplace=True)
        y_pred = self.get_ypredicted_value()
        test_split_idx = int(round(df.shape[0] * (1-self.test_size)))
        y_train, X_train, y_valid, X_valid, y_test, X_test = self.split_data()

        predicted_prices = df.loc[test_split_idx+1:].copy()
        predicted_prices['Close'] = y_pred

        plt.figure(figsize=(15, 8))
        sns.lineplot(y=y_test, x=predicted_prices.Date)
        sns.lineplot(y=y_pred, x=predicted_prices.Date)
        plt.legend(['Predicted', 'Actual'])
        plt.title(f'{self.stock_ticker} XGBoost Model')
        plt.show()


if __name__ == "__main__":
    XGBoostModel(stock_ticker='RNDR-USD').plot_xgboost_prediction()
