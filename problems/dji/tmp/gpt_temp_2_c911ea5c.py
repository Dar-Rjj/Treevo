defined financial logic steps.}

```python
def heuristics_v2(df):
    # Momentum factor: 10-day return
    df['momentum'] = df['close'].pct_change(10)
    
    # Volatility factor: standard deviation of 10-day returns
    df['volatility'] = df['close'].pct_change().rolling(window=10).std()
    
    # Liquidity factor: volume-to-price ratio
    df['liquidity'] = df['volume'] / df['close']
    
    # Combine factors into a single heuristics matrix
    heuristics_matrix = pd.concat([df['momentum'], df['volatility'], df['liquidity']], axis=1)
    heuristics_matrix.columns = ['Momentum', 'Volatility', 'Liquidity']
    
    return heuristics_matrix
