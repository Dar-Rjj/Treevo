defined as volume times close price, with a weighted moving average of returns, aiming to capture both liquidity dynamics and smoothed momentum.}

```python
def heuristics_v2(df):
    # Rate of change in liquidity: Calculate the percentage change in (volume * close) over a 5-day period
    df['liquidity_change'] = (df['volume'] * df['close']).pct_change(periods=5)
    
    # Weighted moving average of daily returns using a 10-day window, with more recent days having higher weights
    weights = pd.Series(range(1, 11))
    df['weighted_returns'] = (df['close'].pct_change().rolling(window=10).apply(lambda x: (x*weights).sum() / weights.sum(), raw=False))
    
    # Combine factors into a single heuristic
    df['heuristic_score'] = df['liquidity_change'] * df['weighted_returns']
    
    # Extract the heuristic score series
    heuristics_matrix = df['heuristic_score']
    
    return heuristics_matrix
