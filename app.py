import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Placeholder model and scaler - replace these with your trained model and scaler
best_model = RandomForestClassifier()  # Load your actual trained model here
scaler = StandardScaler()  # Load your actual trained scaler here

# Function to fetch live data and compute indicators
def fetch_live_data(stock_symbol):
    live_data = yf.download(stock_symbol, period="60d", interval="1d")
    
    # Compute technical indicators
    live_data['SMA_20'] = live_data['Close'].rolling(window=20).mean()
    live_data['SMA_50'] = live_data['Close'].rolling(window=50).mean()
    live_data['RSI'] = compute_rsi(live_data['Close'], 14)
    live_data['EMA_12'] = live_data['Close'].ewm(span=12, adjust=False).mean()
    live_data['EMA_26'] = live_data['Close'].ewm(span=26, adjust=False).mean()
    live_data['MACD'] = live_data['EMA_12'] - live_data['EMA_26']
    live_data['Signal_Line'] = live_data['MACD'].ewm(span=9, adjust=False).mean()
    live_data['BB_Middle'] = live_data['Close'].rolling(window=20).mean()
    live_data['BB_Upper'] = live_data['BB_Middle'] + (2 * live_data['Close'].rolling(window=20).std())
    live_data['BB_Lower'] = live_data['BB_Middle'] - (2 * live_data['Close'].rolling(window=20).std())
    live_data['Volatility'] = live_data['Close'].rolling(window=10).std()
    live_data['Momentum'] = live_data['Close'].diff(10) / live_data['Close'].shift(10)

    # Extract the latest features
    latest_features = live_data.iloc[-1][['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower', 'Volatility', 'Momentum']]
    latest_features_scaled = scaler.transform([latest_features])  # Scale features
    
    # Make prediction
    signal = best_model.predict(latest_features_scaled)[0]
    signal_map = {1: "Buy", -1: "Sell", 0: "Hold"}
    return signal_map[signal], live_data

# Function to compute RSI
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Streamlit app setup
st.set_page_config(page_title="Stock Genie", page_icon="💰", layout="wide")

# Custom styling
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e293b, #a5b4fc);
    font-family: 'Courier New', Courier, monospace;
}
header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #f5f3ff;'>📈 Stock Genie: Gen Z Meets Wall Street</h1>", unsafe_allow_html=True)

# User input for stock symbol
stock_symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")

# Fetch and display prediction
if st.button("Get Prediction"):
    signal, live_data = fetch_live_data(stock_symbol)
    st.markdown(f"### Prediction: **{signal}**")
    
    # Plot stock prices and Bollinger Bands
    plt.figure(figsize=(14, 8))
    plt.plot(live_data['Close'], label="Closing Price", color="cyan")
    plt.fill_between(live_data.index, live_data['BB_Upper'], live_data['BB_Lower'], color="gray", alpha=0.2, label="Bollinger Bands")
    plt.title(f"Stock Price for {stock_symbol}")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    st.pyplot(plt)
