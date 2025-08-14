importance of trading volume in price trends.}

```python
def heuristics_v2(df):
    df['weighted_close'] = df['close'] * df['volume']
    weighted_moving_average = df['weighted_close'].rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    heuristics_matrix = weighted_moving_average.rename('heuristics')
    return heuristics_matrix
