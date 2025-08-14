defined period, then normalizing this spread with a central measure of the closing price.

{The new algorithm calculates a heuristic factor by first determining the ratio of the highest to the lowest stock prices over a 60-day rolling window, then multiplying this ratio by the average trading volume over the same window, aiming to incorporate both price movement and liquidity.}

```python
def heuristics_v2(df):
    df['price_ratio'] = df['high'] / df['low']
    heuristics_matrix = df['price_ratio'].rolling(window=60).mean() * df['volume'].rolling(window=60).mean()
    return heuristics_matrix
