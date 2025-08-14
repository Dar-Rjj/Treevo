importance to recent prices.}

```python
def heuristics_v2(df):
    df['median_price'] = (df['high'] + df['low']) / 2
    weights = [0.1, 0.2, 0.3, 0.2, 0.2]  # Custom weights for the 5-day window
    df['wma_median_price_5'] = df['median_price'].rolling(window=5).apply(lambda x: (x * weights).sum(), raw=True)
    df['ema_volume_10'] = df['volume'].ewm(span=10, adjust=False).mean()
    heuristics_matrix = df['wma_median_price_5'] * df['ema_volume_10']
    return heuristics_matrix
