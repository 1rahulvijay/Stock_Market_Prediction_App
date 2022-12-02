from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib.pyplot import style
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from StockDataAnalyzer import StockDatapipeline
import xgboost as xgb
import seaborn as sns
from xgboost import plot_importance, plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
style.use('ggplot')


class LongShortTermMemory:
    def __init__(self, stock_ticker):
        self.stock_ticker = stock_ticker
        self.data = StockDatapipeline.get_stock_data_from_ticker(stock_ticker)
        self.aim = "Close"
        self.window_len = 5
        self.test_size = 0.2
        self.zero_base = True
        self.lstm_neurons = 50
        self.epochs = 100
        self.batch_size = 32
        self.loss = 'mse'
        self.dropout = 0.24
        self.optimizer = 'adam'

    @staticmethod
    def get_defined_metrics() -> None:
        try:
            defined_metrics = [tf.keras.metrics.MeanSquaredError(name="MSE")]
            return defined_metrics
        except Exception as e:
            self.logger.error(f"cannot retrieve mean squared metrics: {e}")

    @staticmethod
    def get_callback() -> None:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, mode="min", verbose=1
        )
        return callback

    def process_data(self):
        data = self.data
        data = data.set_index('Date')
        data.index = pd.to_datetime(data.index, unit='ns')
        data.sort_index(ascending=True, inplace=True)
        return data

    def get_dynamic_train_test_data(self):
        data = self.process_data()
        row_len = round(len(data.index) * .80)
        train_data = data.iloc[:row_len]
        test_data = data.iloc[row_len:]
        return train_data, test_data

    @staticmethod
    def line_plot(line1, line2, label1=None, label2=None, lw=2, stock_ticker=None):
        fig, ax = plt.subplots(1, figsize=(13, 7))
        ax.plot(line1, label=label1, linewidth=lw)
        ax.plot(line2, label=label2, linewidth=lw)
        ax.set_ylabel(stock_ticker, fontsize=14)
        ax.set_title(
            f'{stock_ticker} LSTM Model', fontsize=16)
        ax.legend(loc='best', fontsize=16)
        plt.show()

    def plot_train_test_split(self):
        train_data, test_data = self.get_dynamic_train_test_data()
        self.line_plot(train_data[self.aim],
                       test_data[self.aim], 'training', 'test', stock_ticker=self.stock_ticker)

    @staticmethod
    def normalise_zero_base(continuous):
        return continuous / continuous.iloc[0] - 1

    @staticmethod
    def normalise_min_max(continuous):
        return (continuous - continuous.min()) / (data.max() - continuous.min())

    def extract_window_data(self, continuous, window_len=5, zero_base=True):
        window_data = []
        for idx in range(len(continuous) - window_len):
            tmp = continuous[idx: (idx + window_len)].copy()
            if zero_base:
                tmp = self.normalise_zero_base(tmp)
            window_data.append(tmp.values)
        return np.array(window_data)

    def prepare_data(self, continuous, aim, window_len=10, zero_base=True, test_size=0.2):
        train_data, test_data = self.get_dynamic_train_test_data()
        X_train = self.extract_window_data(train_data, window_len, zero_base)
        X_test = self.extract_window_data(test_data, window_len, zero_base)
        y_train = train_data[aim][window_len:].values
        y_test = test_data[aim][window_len:].values
        if zero_base:
            y_train = y_train / train_data[aim][:-window_len].values - 1
            y_test = y_test / test_data[aim][:-window_len].values - 1

        return train_data, test_data, X_train, X_test, y_train, y_test

    @staticmethod
    def build_lstm_model(input_data, output_size, neurons, activ_func='linear',
                         dropout=0.2, loss='mse', optimizer='adam'):
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(
            input_data.shape[1], input_data.shape[2])))
        model.add(Dropout(dropout))
        model.add(Dense(units=output_size))
        model.add(Activation(activ_func))

        model.compile(loss=loss, optimizer=optimizer)
        return model

    def training_buidling_lstm_model(self):
        np.random.seed(245)
        train_data, test_data, X_train, X_test, y_train, y_test = self.prepare_data(
            self.data, self.aim, window_len=self.window_len, zero_base=self.zero_base, test_size=self.test_size)

        model = self.build_lstm_model(
            X_train, output_size=1, neurons=self.lstm_neurons, dropout=self.dropout, loss=self.loss,
            optimizer=self.optimizer)
        modelfit = model.fit(
            X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=True, callbacks=self.get_callback())

        return modelfit, model

    def plot_validation_training_loss(self):
        modelfit, model = self.training_buidling_lstm_model()
        plt.plot(modelfit.history['loss'], 'r',
                 linewidth=2, label='Training loss')
        plt.plot(modelfit.history['val_loss'], 'g',
                 linewidth=2, label='Validation loss')
        plt.title(f'LSTM Neural Networks - {self.stock_ticker} Model')
        plt.xlabel('Epochs numbers')
        plt.ylabel('MSE numbers')
        plt.show()

    def test_predicts(self):
        modelfit, model = self.training_buidling_lstm_model()
        train_data, test_data, X_train, X_test, y_train, y_test = self.prepare_data(
            self.data, self.aim, window_len=self.window_len, zero_base=self.zero_base, test_size=self.test_size)
        targets = test_data[self.aim][self.window_len:]
        preds = model.predict(X_test).squeeze()
        mean_absoulte_err = mean_absolute_error(preds, y_test)
        r2_sco = r2_score(y_test, preds)
        r2_sco = r2_sco*100

        print(r2_sco, mean_absoulte_err)
        return preds

    def plot_prediction(self):
        train_data, test_data, X_train, X_test, y_train, y_test = self.prepare_data(
            self.data, self.aim, window_len=self.window_len, zero_base=self.zero_base, test_size=self.test_size)
        preds = self.test_predicts()
        targets = test_data[self.aim][self.window_len:]
        preds = test_data[self.aim].values[:-self.window_len] * (preds + 1)
        preds = pd.Series(index=targets.index, data=preds)
        self.line_plot(targets, preds, 'actual', 'prediction',
                       lw=4, stock_ticker=self.stock_ticker)
        # plt.show()


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
