defined as the sum of high and low prices divided by volume, over a 20-day window, to forecast future returns.}

```python
def heuristics_v2(df):
    df['momentum'] = df['close'].pct_change(periods=20)
    df['liquidity_measure'] = (df['high'] + df['low']) / df['volume']
    heuristics_matrix = df['momentum'] * df['liquidity_measure'].rolling(window=20).mean()
    return heuristics_matrix
