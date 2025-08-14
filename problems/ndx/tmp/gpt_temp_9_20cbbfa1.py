defined as the sum of the open and close prices weighted by their respective volumes, then applying a 20-day exponentially weighted moving average to smooth the result.}
```python
def heuristics_v2(df):
    avg_price = (df['high'] + df['low']) / 2
    vol_adjusted_price = (df['open'] * df['volume'] + df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    heuristic_values = avg_price - vol_adjusted_price
    heuristics_matrix = heuristic_values.ewm(span=20, adjust=False).mean().dropna()
    return heuristics_matrix
