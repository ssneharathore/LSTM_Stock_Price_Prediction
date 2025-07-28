import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.preprocessing import prepare_data
from scripts.lstm_model import build_model
from scripts.evaluate import evaluate_model
import datetime

st.set_page_config(page_title="📈 Stock Price Prediction", layout="wide")

st.title("📊 LSTM Stock Price Prediction with MACD & RSI")

# Sidebar Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker", value="RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Fetch Data
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.dropna(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data found. Try a different ticker.")
    st.stop()

# Compute Indicators
exp1 = data['Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = exp1 - exp2
data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Plot Technical Indicators
with st.expander("📉 Technical Indicators (MACD & RSI)"):
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axs[0].plot(data['Close'], label='Close Price')
    axs[0].set_title(f'{ticker} Closing Price')
    axs[0].legend(); axs[0].grid()

    axs[1].plot(data['MACD'], label='MACD', color='blue')
    axs[1].plot(data['Signal'], label='Signal', color='orange')
    axs[1].set_title('MACD & Signal Line')
    axs[1].legend(); axs[1].grid()

    axs[2].plot(data['RSI'], label='RSI', color='purple')
    axs[2].axhline(70, color='red', linestyle='--')
    axs[2].axhline(30, color='green', linestyle='--')
    axs[2].set_title('RSI')
    axs[2].legend(); axs[2].grid()

    plt.tight_layout()
    st.pyplot(fig)

# Buy/Sell signals (based on MACD & RSI)
data['Buy'] = (data['MACD'] > data['Signal']) & (data['RSI'] < 30)
data['Sell'] = (data['MACD'] < data['Signal']) & (data['RSI'] > 70)

# Train Model
st.subheader("🧠 Predicting Stock Prices with LSTM")

X, y, scaler = prepare_data(data)
model = build_model((X.shape[1], X.shape[2]))
model.fit(X, y, epochs=10, batch_size=32, verbose=0)

X_test, y_test = X[-100:], y[-100:]
rmse, mae, predictions, y_actual = evaluate_model(model, X_test, y_test, scaler)

# Plot Predictions
st.markdown(f"**RMSE:** {rmse:.2f} | **MAE:** {mae:.2f}")

pred_dates = data.index[-len(y_test):]

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(pred_dates, y_actual, label='Actual Price')
ax2.plot(pred_dates, predictions, label='Predicted Price')
ax2.set_title("LSTM Stock Price Prediction")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
ax2.grid()
st.pyplot(fig2)

# Show Buy/Sell markers
with st.expander("💡 Buy/Sell Suggestions (from MACD & RSI)"):
    signals = data.loc[data['Buy'] | data['Sell'], ['Close', 'Buy', 'Sell']]
    st.dataframe(signals.tail(10))
