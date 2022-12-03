import streamlit as st
import pandas as pd
import numpy as np
import plost
from PIL import Image
from StockDataAnalyzer import StockDatapipeline

# Page setting
st.set_page_config(layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    # button = """<div class="row-widget stButton" style="width: 64px;">
    # <button kind="primary" class="css-4eonon edgvbvh1">
    # </button></div>"""
    st.markdown(hide_st_style, unsafe_allow_html=True)
    # st.markdown(button, unsafe_allow_html=True)


stock_list = ['GOOGL', 'BABA', 'NFLX', 'AAPL', 'MSFT', 'META', 'AMZN']
# Row A
# with st.sidebar.container():
#     image = Image.open('stocks.jpg')
#     st.image(image, width=1, use_column_width=True)




with st.sidebar.header("Stock List"):
    stock = st.sidebar.selectbox(label='Stocks',
                                 options=stock_list)

curr_price = StockDatapipeline(stock_ticker=stock).get_realtime_stock_price()
curr_open = StockDatapipeline(
    stock_ticker=stock).get_current_stock_open_price()
curr_vol = StockDatapipeline(stock_ticker=stock).get_stock_volume()
b1, b2, b3 = st.columns(3)
b1.metric("Stock Price", str(curr_price)+' USD')
b2.metric("Open", str(curr_open)+' USD')
b3.metric("Volume", str(curr_vol)+' USD')

# Row C
with st.sidebar:
    st.button("Plot SMA", kwargs={
        'clicked_button_ix': 1, 'n_buttons': 4
    })
    st.button("Plot trend", kwargs={
        'clicked_button_ix': 2, 'n_buttons': 4
    })

    st.button("Plot Y", kwargs={
        'clicked_button_ix': 3, 'n_buttons': 4

    })

    st.button("Plot S", kwargs={
        'clicked_button_ix': 4, 'n_buttons': 4
    })


c1, c2 = st.columns((5, 5))
with c1:
    st.markdown('### Heatmap')
    st.pyplot(fig=StockDatapipeline(stock_ticker=stock).plot_100_days_sma())
with c2:
    st.markdown('### Bar chart')
    st.pyplot(fig=StockDatapipeline(stock_ticker=stock).plot_200_days_sma())

d1, d2 = st.columns((5, 5))
with d1:
    st.markdown('### Heatmap')
    st.pyplot(fig=StockDatapipeline(stock_ticker=stock).plot_two_side_view())
with d2:
    st.markdown('### Bar chart')
    st.pyplot(fig=StockDatapipeline(
        stock_ticker=stock).plot_stock_ticker_data())

with st.container():
    st.write("This is inside the container")
    st.pyplot(fig=StockDatapipeline(
        stock_ticker=stock).plot_week_and_month_moving_average())
