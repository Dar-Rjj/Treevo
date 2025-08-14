import pandas as pd

def heuristics_v2(df):
    # Calculate short-term and long-term moving averages
    short_moving_avg = df['close'].rolling(window=10).mean()
    long_moving_avg = df['close'].rolling(window=50).mean()
    
    # Momentum indicator
    momentum = df['close'] - df['close'].shift(12)
    
    # Volatility (standard deviation over a 20 day window)
    volatility = df['close'].rolling(window=20).std()
    
    # Heuristic factor: difference between short and long moving averages
    moving_avg_diff = short_moving_avg - long_moving_avg
    
    # Combine factors into a single heuristic matrix
    heuristics_matrix = pd.concat([moving_avg_diff, momentum, volatility], axis=1)
    heuristics_matrix.columns = ['Moving_Avg_Diff', 'Momentum', 'Volatility']
    
    return heuristics_matrix
```

Note: The function return heuristics_matrix
