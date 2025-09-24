import pandas as pd

AAPL = pd.read_csv('AAPL.csv')
AAPL.index = pd.to_datetime(AAPL['Date'])
AAPL = AAPL.drop(columns=['Date'])
AAPL.to_csv('AAPL.csv')
print(AAPL.head())


# pd read csv 
df = pd.read_csv('AAPL.csv')
df.index = pd.to_datetime(df['Date'])
df = df.drop(columns=['Date'])
print(df.head())

# pd read csv with index
df = pd.read_csv('AAPL.csv', index_col='Date')
print(df.head())

# pd read csv with index
df = pd.read_csv('AAPL.csv', index_col='Date')
print(df.head())

# plot and  basic statistics
df.plot()
df.describe()