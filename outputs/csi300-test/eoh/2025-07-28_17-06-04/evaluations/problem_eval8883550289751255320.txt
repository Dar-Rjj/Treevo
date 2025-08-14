defined as the average of the lowest lows over the past 20 days, to identify potential buying pressure.}

```python
def heuristics_v2(df):
    support_level = df['low'].rolling(window=20).min()
    heuristics_matrix = (df['close'] - support_level) / support_level
    return heuristics_matrix
