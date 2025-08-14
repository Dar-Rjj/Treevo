defined by the exponential moving average of the high and low prices over a specified period.}

```python
def heuristics_v2(df):
    df['EMA_High'] = df['high'].ewm(span=10, adjust=False).mean()
    df['EMA_Low'] = df['low'].ewm(span=10, adjust=False).mean()
    heuristics_matrix = (df['close'] - df['EMA_Low']) / (df['EMA_High'] - df['EMA_Low'])
    return heuristics_matrix
