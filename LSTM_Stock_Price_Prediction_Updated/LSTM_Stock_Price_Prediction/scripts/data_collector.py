import yfinance as yf
import datetime

def fetch_data(ticker="AAPL", start="2015-01-01", end=None):
    if end is None:
        end = datetime.datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]
