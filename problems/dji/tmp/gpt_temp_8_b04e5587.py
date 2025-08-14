importance to very recent trading activity.}
```python
def heuristics_v2(df):
    # Apply an exponentially increasing weight to the volume
    df['adjusted_volume'] = df['volume'] * (1.5 ** (df.index.to_series().rank(ascending=True) - 1))
    
    # Calculate the 20-day exponential moving average of the adjusted volume
    df['volume_ema_adjusted'] = df['adjusted_volume'].ewm(span=20, adjust=False).mean()
    
    # Compute the heuristic factor
    heuristics_matrix = df['close'] / df['volume_ema_adjusted']
    
    return heuristics_matrix
