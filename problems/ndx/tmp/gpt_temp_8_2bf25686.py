defined as the weighted sum of open and close prices based on their relative volumes, then applying a 10-day exponentially weighted moving average to smooth the result.}

```python
def heuristics_v2(df):
    avg_price = (df['high'] + df['low']) / 2
    weight_open = df['volume'].shift(1) / (df['volume'].shift(1) + df['volume'])
    weight_close = 1 - weight_open
    vol_adj_price = (df['open'] * weight_open + df['close'] * weight_close)
    heuristic_values = avg_price / vol_adj_price
    heuristics_matrix = heuristic_values.ewm(span=10, adjust=False).mean().dropna()
    return heuristics_matrix
