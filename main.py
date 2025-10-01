import yfinance as yf
import pandas as pd
import numpy as np
sp500_ticker = '^GSPC'
sp500_data = yf.download(sp500_ticker, start='2015-09-01', end='2017-09-01')
sp500_data.head()
print(sp500_data.columns)
if 'Adj Close' not in sp500_data.columns:
    sp500_data['Adj Close'] = sp500_data['Close']
# Drop unnecessary columns
sp500_data.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)
print(sp500_data.head())



# Calculate daily returns
sp500_data['Daily_Returns'] = sp500_data['Adj Close'].pct_change()

# Calculate the simple moving average (SMA) as a technical indicator
sp500_data['SMA_50'] = sp500_data['Adj Close'].rolling(window=50).mean()
sp500_data['SMA_200'] = sp500_data['Adj Close'].rolling(window=200).mean()

# Drop the first 200 rows as they contain NaN values due to the SMA calculation
sp500_data.dropna(inplace=True)

# Let's inspect the data again
sp500_data.head()


# Generate a trading signal
sp500_data['Signal'] = 0.0
sp500_data['Signal'][50:] = np.where(sp500_data['SMA_50'][50:] > sp500_data['SMA_200'][50:], 1.0, 0.0)

# Create a trading position based on the signal
sp500_data['Position'] = sp500_data['Signal'].diff()
sp500_data.head(20)


# Calculate strategy returns
sp500_data['Strategy_Returns'] = sp500_data['Daily_Returns'] * sp500_data['Signal'].shift(1)

# Calculate cumulative returns for both the S&P 500 and the strategy
sp500_data['SP500_Cumulative_Returns'] = (1 + sp500_data['Daily_Returns']).cumprod()
sp500_data['Strategy_Cumulative_Returns'] = (1 + sp500_data['Strategy_Returns']).cumprod()

# Plot the cumulative returns to visualize performance
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(sp500_data['SP500_Cumulative_Returns'], label='S&P 500 Buy & Hold')
plt.plot(sp500_data['Strategy_Cumulative_Returns'], label='SMA Crossover Strategy')
plt.title('S&P 500 vs. SMA Crossover Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

# Calculate alpha and other key metrics
# A common way to calculate alpha is using regression, but for simplicity, we can also look at the difference in returns.
final_strategy_return = sp500_data['Strategy_Cumulative_Returns'].iloc[-1] - 1
final_sp500_return = sp500_data['SP500_Cumulative_Returns'].iloc[-1] - 1
alpha = final_strategy_return - final_sp500_return
print(f'Alpha: {alpha:.2f}')