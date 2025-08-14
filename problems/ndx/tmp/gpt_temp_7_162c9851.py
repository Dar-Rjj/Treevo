defined as the difference between the 50-day and 200-day Exponential Moving Averages (EMAs) of the closing price, adjusted by a weighted moving average of the past 10 days' closing prices.}

```python
def heuristics_v2(df):
    # Calculate 50-day and 200-day EMA for the closing price
    ema_50 = df['close'].ewm(span=50, adjust=False).mean()
    ema_200 = df['close'].ewm(span=200, adjust=False).mean()

    # Calculate the 20-day average volume
    avg_volume_20 = df['volume'].rolling(window=20).mean()

    # Calculate the ratio of the current day's volume to the 20-day average volume
    volume_ratio = df['volume'] / avg_volume_20

    # Calculate the 10-day weighted moving average of the closing prices
    wma_10 = df['close'].ewm(span=10, adjust=False).mean()

    # Combine the EMA difference, volume ratio, and the 10-day WMA
    ema_diff = ema_50 - ema_200
    heuristics_matrix = (ema_diff * wma_10 + volume_ratio) / 2

    return heuristics_matrix
