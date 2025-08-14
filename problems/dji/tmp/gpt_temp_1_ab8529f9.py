defined as (high + low) / 2 * sqrt(volume), and then computes the difference between its 10-day and 30-day exponential moving averages.}

```python
def heuristics_v2(df):
    df['adjusted_price'] = (df['high'] + df['low']) / 2 * df['volume'].apply(np.sqrt)
    ewma_10 = df['adjusted_price'].ewm(span=10).mean()
    ewma_30 = df['adjusted_price'].ewm(span=30).mean()
    heuristics_matrix = ewma_10 - ewma_30
    return heuristics_matrix
