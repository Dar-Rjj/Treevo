defined as the percentage change in closing price over a 20-day period, and adjusting this measure by subtracting the average of the positive daily returns over the same period to better capture the underlying trend.}

```python
def heuristics_v2(df):
    # Calculate the 20-day momentum
    momentum = df['close'].pct_change(periods=20)
    # Filter for positive daily returns
    positive_returns = df['close'].pct_change().where(lambda x: x > 0, 0)
    # Calculate the average of positive daily returns over 20 days
    avg_positive_returns = positive_returns.rolling(window=20).mean()
    # Adjust the momentum by the average of positive daily returns
    heuristics_matrix = momentum - avg_positive_returns
    return heuristics_matrix
