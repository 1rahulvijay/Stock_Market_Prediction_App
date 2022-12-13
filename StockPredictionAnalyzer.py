import numpy as np
import pandas as pd
import tensorflow as tf
from StockDataAnalyzer import StockDatapipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
from tensorflow.python.keras.layers import SimpleRNNCell
from tensorflow.python.keras.layers import RNN
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from matplotlib.pyplot import style
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib as mpl
from xgboost import plot_importance, plot_tree
style.use('dark_background')
COLOR = 'White'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR


class RNN_Model:
    def __init__(self, stock_ticker):
        self.stock_ticker = stock_ticker
        self.aim = "Close"
        self.scaler = MinMaxScaler()
        self.rnn_neurons = 256
        self.epochs = 100
        self.batch_size = 32
        self.loss = 'mse'
        self.dropout = 0.24
        self.optimizer = 'adam'
        self.data, self.y = self.__get_data()
        self.row_len = round(len(self.data.index) * 0.95)
        self.X_train, self.y_train, self.X_test, self.y_test = self.__scaler_transform()
        self.X_train = np.expand_dims(self.X_train, axis=1)
        self.X_test = np.expand_dims(self.X_test, axis=1)

    def __get_data(self):
        data = StockDatapipeline.get_stock_data_from_ticker(self.stock_ticker)
        data = data[["Date", "Volume", "Open", "Close"]]
        y = data.loc[:, ['Close', 'Date']]
        data = data.drop(['Close'], axis='columns')
        y = y.set_index('Date')
        y.index = pd.to_datetime(y.index, unit='ns')
        data = data.set_index('Date')
        data.index = pd.to_datetime(data.index, unit='ns')
        # print(y)
        return data, y

    @staticmethod
    def get_callback() -> None:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, mode="min", verbose=1
        )
        return callback

    def __get_dynamic_train_test_data(self):
        X_train = self.data[:self.row_len]
        X_test = self.data[self.row_len:]
        y_train = self.y[:self.row_len]
        y_test = self.y[self.row_len:]
        #print(X_train, y_train, X_test, y_test)
        return X_train, y_train, X_test, y_test

    def __scaler_transform(self):
        X_train, y_train, X_test, y_test = self.__get_dynamic_train_test_data()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.fit_transform(X_test)
        y_train = self.scaler.fit_transform(y_train)
        y_test = self.scaler.fit_transform(y_test)
        #print(X_train, y_train, X_test, y_test)

        return X_train, y_train, X_test, y_test

    @staticmethod
    def line_plot(line1, line2, label1=None, label2=None, lw=2, stock_ticker=None, title=None):
        fig, ax = plt.subplots(1, figsize=(7, 3))
        ax.plot(line1, label=label1, linewidth=lw, color='tab:Red')
        ax.plot(line2, label=label2, linewidth=lw, color='#74b9ff')
        ax.set_ylabel("Close", fontsize=12)
        ax.set_title(
            f'{stock_ticker} {title} Model', fontsize=10)
        ax.legend(loc='best', fontsize=10)
        fig.autofmt_xdate()
        # plt.show()
        return fig

    @staticmethod
    def build_RNN_model(input_data, output_size, neurons, activ_func='tanh',
                        dropout=0.21, loss='mse', optimizer='adam'):
        """Recurrent Neural Network Model"""
        model = Sequential()
        model.add(RNN(cell=[SimpleRNNCell(256),
                            SimpleRNNCell(512),
                            SimpleRNNCell(1024)], input_shape=(1, 2)))
        model.add(Dropout(dropout))
        model.add(Dense(units=64*4))
        model.add(Activation("relu"))
        model.add(Dropout(dropout))
        model.add(Dense(units=output_size))
        model.add(Activation(activ_func))

        model.compile(loss=loss, optimizer=optimizer)
        return model

    def train_rnn_model(self):
        np.random.seed(1024)
        model = self.build_RNN_model(self.X_train, output_size=1, neurons=self.rnn_neurons,
                                     dropout=self.dropout, loss=self.loss, optimizer=self.optimizer)

        modelfit = model.fit(self.X_train, self.y_train, validation_data=(
            self.X_test, self.y_test), epochs=self.epochs, batch_size=self.batch_size,
            verbose=1, shuffle=True, callbacks=self.get_callback())

        return model, modelfit

    def plot_rnn_prediction(self):
        model, modelfit = self.train_rnn_model()
        test_data = self.y[self.row_len:]
        preds = model.predict(self.X_test).squeeze()

        predd = self.scaler.inverse_transform(preds.reshape(len(test_data), 1))
        predd = pd.Series(index=test_data.index, data=predd.flatten())
        return self.line_plot(test_data.Close, predd, 'Actual',
                              'Predicted', stock_ticker=self.stock_ticker, title='RNN')


class LSTM_Model(RNN_Model):
    def __init__(self, stock_ticker):
        super().__init__(stock_ticker)
        self.window_len = 5
        self.test_size = 0.2
        self.zero_base = True
        self.lstm_neurons = 256
        self.epochs = 100
        self.batch_size = 32
        self.loss = 'mse'
        self.dropout = 0.24
        self.optimizer = 'adam'

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

    def train_lstm_model(self):
        model = self.build_lstm_model(
            self.X_train, output_size=1, neurons=self.lstm_neurons, dropout=self.dropout, loss=self.loss,
            optimizer=self.optimizer)
        modelfit = model.fit(self.X_train, self.y_train, validation_data=(
            self.X_test, self.y_test), epochs=self.epochs,
            batch_size=self.batch_size, verbose=1, shuffle=True, callbacks=super(LSTM_Model, self).get_callback())

        return model, modelfit

    def plot_lstm_prediction(self):
        model, modelfit = self.train_lstm_model()
        test_data = self.y[self.row_len:]
        preds = model.predict(self.X_test).squeeze()

        predd = self.scaler.inverse_transform(preds.reshape(len(test_data), 1))
        predd = pd.Series(index=test_data.index, data=predd.flatten())
        return super(LSTM_Model, self).line_plot(test_data.Close, predd, 'Actual',
                                                 'Predicted', stock_ticker=self.stock_ticker, title='LSTM')


class XGBoostModel:
    def __init__(self, stock_ticker):
        self.test_size = 0.05
        self.valid_size = 0.05
        self.stock_ticker = stock_ticker
        self.data = StockDatapipeline.get_stock_data_from_ticker(stock_ticker)
        self.gamma = 0.001
        self.learning_rate = 0.05
        self.max_depth = 12
        self.n_estimators = 400
        self.random_state = 42

    def get_data(self):
        df = self.data
        df = df[["Date", "Volume", "Open", "Close"]]
        return df

    def split_data(self):
        drop_cols = ['Date']
        df = self.get_data()
        df.reset_index(inplace=True)
        # print(df)

        test_split_idx = int(round(df.shape[0] * (1-self.test_size)))
        valid_split_idx = int(round(
            df.shape[0] * (1-(self.valid_size+self.test_size))))

        # print(test_split_idx)
        # print(valid_split_idx)

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

        #print(y_train, X_train, y_valid, X_valid, y_test, X_test)
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
        return y_pred

    def get_ypredicted_value(self):
        y_train, X_train, y_valid, X_valid, y_test, X_test = self.split_data()
        model = self.build_and_train_model()
        y_pred = self.y_pred(model=model, X_test=X_test)
        # print(y_pred)
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

        fig = plt.figure(figsize=(7, 3))
        sns.lineplot(x=predicted_prices.Date, y=y_test, color='tab:Red')
        sns.lineplot(x=predicted_prices.Date, y=y_pred, color='#74b9ff')
        plt.legend(['Actual', 'Predicted'], loc='best')
        plt.title(f'{self.stock_ticker} XGBoost Model')
        fig.autofmt_xdate()

        return fig


if __name__ == "__main__":
    XGBoostModel(stock_ticker='IBM').plot_xgboost_prediction()
    # RNN_Model(stock_ticker='NFLX').plot_rnn_prediction()
    # LSTM_Model(stock_ticker='NFLX').plot_lstm_prediction()
