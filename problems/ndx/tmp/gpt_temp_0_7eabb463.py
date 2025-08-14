import pandas as pd
    
    # Calculate the daily return
    daily_return = df['close'].pct_change()
    
    # Calculate the difference between close and volume
    diff_close_volume = df['close'] - df['volume']
    
    # Calculate the 5-day moving average of the daily return for weight adjustment
    rolling_avg_return = daily_return.rolling(window=5).mean().fillna(0)
    
    # Lag the daily return by 1 period to avoid look-ahead bias
    lag_daily_return = daily_return.shift(1).fillna(0)
    
    # Compute the weighted difference
    weighted_diff = diff_close_volume * (rolling_avg_return + 1)
    
    # Generate the final heuristic matrix combining weighted differences and lagged returns
    heuristics_matrix = weighted_diff + lag_daily_return
    
    return heuristics_matrix
```
Note: The function `heuristics_v2` is designed to return heuristics_matrix
