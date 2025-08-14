importance of recent price changes and an average level of trading activity.}

```python
def heuristics_v2(df):
    df['ema_close_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['sma_volume_20'] = df['volume'].rolling(window=20).mean()
    heuristics_matrix = df['ema_close_10'] * df['sma_volume_20']
    return heuristics_matrix
