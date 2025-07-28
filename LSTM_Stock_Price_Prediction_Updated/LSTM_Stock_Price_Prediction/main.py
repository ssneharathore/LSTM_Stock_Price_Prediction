from scripts.data_collector import fetch_data
from scripts.preprocessing import prepare_data
from scripts.lstm_model import build_model
from scripts.evaluate import evaluate_model
import matplotlib.pyplot as plt
import ta
import pandas as pd

# 1. Dynamic Ticker Input
ticker = input("Enter stock ticker (e.g., AAPL, MSFT, INFY.NS): ")

# 2. Fetch Data
print("Fetching stock data...")
data = fetch_data(ticker, start="2015-01-01")

# 3. Optional: Visualize MACD and RSI
df = data.copy()

# ✅ MACD (manual calculation to avoid ta errors)
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# ✅ RSI manual implementation
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14, min_periods=14).mean()
avg_loss = loss.rolling(window=14, min_periods=14).mean()

rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Clean NaNs
df.dropna(inplace=True)

# 4. Add Buy/Sell Annotations
buy_signals = (df['MACD'] > df['MACD_signal']) & (df['RSI'] < 30)
sell_signals = (df['MACD'] < df['MACD_signal']) & (df['RSI'] > 70)

buy_dates = df.index[buy_signals]
buy_prices = df['Close'][buy_signals]
sell_dates = df.index[sell_signals]
sell_prices = df['Close'][sell_signals]

# 📊 Plot indicators
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(df['Close'], label='Close Price', color='blue')
plt.scatter(buy_dates, buy_prices, marker='^', color='green', label='Buy Signal', zorder=5)
plt.scatter(sell_dates, sell_prices, marker='v', color='red', label='Sell Signal', zorder=5)
plt.title(f'{ticker} - Close Price with Buy/Sell Signals')
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(df['MACD'], label='MACD', color='blue')
plt.plot(df['MACD_signal'], label='Signal Line', color='orange')
plt.title(f'{ticker} - MACD')
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(df['RSI'], label='RSI', color='purple')
plt.axhline(70, color='red', linestyle='--', label='Overbought')
plt.axhline(30, color='green', linestyle='--', label='Oversold')
plt.title(f'{ticker} - RSI')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# 5. Preprocessing for LSTM
print("Preprocessing data...")
print("Date range:", data.index.min(), "to", data.index.max())
X, y, scaler = prepare_data(data)  # ✅ Prepares with indicators if included in preprocessing.py

# 6. Build Model
print("Building model...")
model = build_model((X.shape[1], X.shape[2]))

# 7. Train Model
print("Training model...")
model.fit(X, y, epochs=10, batch_size=32)

# 8. Evaluate Model
print("Evaluating model...")
X_test, y_test = X[-100:], y[-100:]
rmse, mae, predictions, y_actual = evaluate_model(model, X_test, y_test, scaler)

# 9. Get Real Dates
dates = data.index[-len(y_test):]

# 10. Plot Prediction Results
plt.figure(figsize=(12, 6))
plt.plot(dates, y_actual, label='Actual Price')
plt.plot(dates, predictions, label='Predicted Price')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"Stock Price Prediction for {ticker}")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()