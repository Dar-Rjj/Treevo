defined as the percentage change in closing prices.}

```python
def heuristics_v2(df):
    daily_return = df['close'].pct_change()
    ema_10 = daily_return.ewm(span=10, adjust=False).mean()
    ema_30 = daily_return.ewm(span=30, adjust=False).mean()
    heuristics_matrix = ema_10 - ema_30
    return heuristics_matrix
