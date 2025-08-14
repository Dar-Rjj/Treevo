defined as the average of the last 5 days' return, and volatility is the standard deviation of returns over the last 20 days.}

```python
def heuristics_v2(df):
    df['Return'] = df['close'].pct_change()
    momentum = df['Return'].rolling(window=5).mean()
    volatility = df['Return'].rolling(window=20).std()
    heuristics_matrix = (momentum * 0.6) + (volatility * 0.4)
    return heuristics_matrix
