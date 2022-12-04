import streamlit as st
from PIL import Image
from StockDataAnalyzer import StockDatapipeline
from StockNewsAnalyzer import StockNews, StockTweets
from StockPredictionAnalyzer import LongShortTermMemory, XGBoostModel
st.set_page_config(layout="wide")


class StockApp:
    def __init__(self):
        self.path = StockDatapipeline.get_current_dir()
        self.settings = StockDatapipeline.load_settings(self.path)
        self.css = self.settings['css_file']
        self.local_css(self.css)
        self.stock_list = self.settings['stock_list']

    @staticmethod
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    def layout(self):

        stock = st.sidebar.selectbox(label='Available Stocks',
                                     options=self.stock_list)

        curr_price = StockDatapipeline(
            stock_ticker=stock).get_realtime_stock_price()
        curr_open = StockDatapipeline(
            stock_ticker=stock).get_current_stock_open_price()
        curr_vol = StockDatapipeline(stock_ticker=stock).get_stock_volume()

        b1, b2, b3 = st.columns(3)
        b1.metric("Stock Price", str(curr_price)+' USD')
        b2.metric("Open", str(curr_open)+' USD')
        b3.metric("Volume", str(float(curr_vol))+' USD')

        with st.sidebar:
            col1, col3 = st.columns(2)
            with col1:
                if st.button(label='Download Stock Data'):
                    StockDatapipeline(
                        stock_ticker=stock).download_stock_ticker_data()
                    st.write('## Stock Data Downloaded')

            with col3:
                if st.button(label="Download Stock Info",):
                    StockDatapipeline(
                        stock_ticker=stock).download_market_info()
                    st.write('## Market Info Downloaded')

        # with st.container():
        #     st.markdown('### Stock Price Chart')
        #     st.pyplot(fig=StockDatapipeline(
        #         stock_ticker=stock).plot_stock_ticker_data())

        with st.container():
            col4, col5, col6 = st.columns(3)
            with col4:
                if st.button(label='50D SMA'):
                    with st.container():
                        st.markdown('## 50 Day Moving Average')
                        st.pyplot(fig=StockDatapipeline(
                            stock_ticker=stock).plot_50_days_sma())

            with col5:
                if st.button(label="100D SMA"):
                    with st.container():
                        st.markdown('## Generating 100 Day Moving Average')
                        st.pyplot(fig=StockDatapipeline(
                            stock_ticker=stock).plot_100_days_sma())

            with col6:
                if st.button(label="200D SMA"):
                    with st.container():
                        st.markdown('## Generating 200 Day Moving Average')
                        st.pyplot(fig=StockDatapipeline(
                            stock_ticker=stock).plot_100_days_sma())
        with st.container():
            col7, col8, col9 = st.columns(3)
            with col7:
                if st.button(label='Relative strength index'):
                    with st.container():
                        st.markdown('## Generating Relative Strength Index')
                        st.pyplot(fig=StockDatapipeline(
                            stock_ticker=stock).plot_rsi())

            with col8:
                if st.button(label="MACD"):
                    with st.container():
                        st.markdown('## Generating MACD Index')
                        st.pyplot(fig=StockDatapipeline(
                            stock_ticker=stock).plot_macd())

            with col9:
                if st.button(label="Week Month SMA"):
                    with st.container():
                        st.markdown('## Generating Moving Average Combined ')
                        st.pyplot(fig=StockDatapipeline(
                            stock_ticker=stock).plot_week_and_month_moving_average())

        with st.container():
            st.markdown('### Weekly and Monthly Moving Average')
            st.pyplot(fig=StockDatapipeline(
                stock_ticker=stock).plot_trend())

        d1, d2 = st.columns((5, 5))
        with d1:
            st.markdown('### Tweets Sentiment Overview')
            st.pyplot(fig=StockNews(
                stock_ticker=stock).plot_tweet_sentiment_donut_chart())
        with d2:
            st.markdown('### Stock Two Side View')
            st.pyplot(fig=StockDatapipeline(
                stock_ticker=stock).plot_two_side_view())

        with st.container():
            st.markdown('### Daily News Sentiments')
            st.pyplot(fig=StockNews(
                stock_ticker=stock).plot_daily_sentiment_barchart())
            # StockTweets(stock_ticker=stock).plot_tweet_sentiment_donut_chart()

        with st.container():
            st.markdown('### Daily News Affecting Price')
            st.pyplot(fig=StockNews(
                stock_ticker=stock).plot_sentiments_with_price())

        with st.container():
            st.markdown('### LSTM Predictions')
            st.pyplot(fig=LongShortTermMemory(
                stock_ticker=stock).plot_prediction())

        with st.container():
            st.markdown('### XGBoost Predictions')
            st.pyplot(fig=XGBoostModel(
                stock_ticker=stock).plot_xgboost_prediction())


if __name__ == "__main__":
    StockApp().layout()
