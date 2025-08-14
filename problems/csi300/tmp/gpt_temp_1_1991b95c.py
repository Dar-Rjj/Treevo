importance to recent data.}

```python
def heuristics_v2(df):
    def calculate_momentum(row, window=10):
        return (row['close'] - df['close'].shift(window).iloc[row.name]) / df['close'].shift(window).iloc[row.name]

    def calculate_volatility(row, window=10):
        returns = df['close'].pct_change().rolling(window=window).std()
        return returns.iloc[row.name]
    
    weights = [i/sum(range(1, 11)) for i in range(1, 11)]
    df['momentum'] = df.apply(calculate_momentum, axis=1)
    df['volatility'] = df.apply(calculate_volatility, axis=1)

    df['heuristic_factor'] = df['momentum'] * (1 - df['volatility'])
    heuristics_matrix = df['heuristic_factor'].rolling(window=10).apply(lambda x: (x*weights).sum(), raw=False)
    return heuristics_matrix
