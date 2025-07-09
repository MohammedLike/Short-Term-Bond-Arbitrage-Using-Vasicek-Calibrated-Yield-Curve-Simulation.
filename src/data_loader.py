import yfinance as yf
import pandas as pd

def get_short_rate_data(ticker='^IRX', start='2015-01-01', end='2024-12-31'):
    data = yf.download(ticker, start=start, end=end)
    if 'Adj Close' in data.columns:
        data = data['Adj Close'] / 100
    elif 'Close' in data.columns:
        data = data['Close'] / 100
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' found in data columns: " + str(data.columns))
    data = data.dropna()
    data.name = "ShortRate"
    return data

