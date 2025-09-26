import pandas as pd
import yfinance as yf

ticker = yf.Ticker("AAPL")
data = ticker.history(period="1d")
print(data)

data.to_csv("AAPL.csv")

tmp = pd.read_csv("AAPL.csv")
print(tmp)

tmp.index = pd.to_datetime(tmp['Date'])
tmp = tmp.drop(columns=['Date'])
print(tmp)

tmp.to_csv("AAPL.csv")
print(tmp)