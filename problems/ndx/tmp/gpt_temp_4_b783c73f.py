defined window, then divides this by the average trading volume over the same period to derive a volatility-based factor.}
```python
def heuristics_v2(df):
    window = 30
    
    high_low_diff = df['high'].rolling(window=window).max() - df['low'].rolling(window=window).min()
    avg_volume = df['volume'].rolling(window=window).mean()
    heuristics_matrix = (high_low_diff / avg_volume).dropna()
    
    return heuristics_matrix
