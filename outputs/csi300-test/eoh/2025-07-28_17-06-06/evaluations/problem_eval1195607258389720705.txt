defined as the difference between the 10-day and 30-day simple moving averages of the closing price, to capture the trend strength.}

```python
def heuristics_v2(df):
    df['10d_sma'] = df['close'].rolling(window=10).mean()
    df['30d_sma'] = df['close'].rolling(window=30).mean()
    df['momentum_factor'] = df['10d_sma'] - df['30d_sma']
    heuristics_matrix = df['momentum_factor'].dropna()
    return heuristics_matrix
