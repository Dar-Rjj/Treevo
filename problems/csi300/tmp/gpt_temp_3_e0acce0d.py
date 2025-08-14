defined period, then combine these measures with another metric (like volume) to create an alpha factor that reflects both price and market activity characteristics.

{The new algorithm calculates an alpha factor by first identifying the days within a 30-day window where the closing price is higher than the opening price, then summing up the daily volume for such positive days, and finally dividing this sum by the total volume over the same period to gauge the relative buying pressure.}

```python
def heuristics_v2(df):
    positive_days = df['close'] > df['open']
    volume_on_positive_days = df[positive_days]['volume'].rolling(window=30).sum()
    total_volume = df['volume'].rolling(window=30).sum()
    heuristics_matrix = volume_on_positive_days / total_volume
    return heuristics_matrix
