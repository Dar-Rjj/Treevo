import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 10-day raw momentum using Close prices
    momentum_10d = df['close'] - df['close'].shift(10)
    
    # Compute 20-day average high-low range as volatility proxy
    daily_range = df['high'] - df['low']
    avg_range_20d = daily_range.rolling(window=20, min_periods=10).mean()
    
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    
    # Apply asymmetric weighting
    # Initialize weights array
    weights = np.ones(len(df))
    
    for i in range(len(df)):
        if i < 20:  # Ensure we have enough data for volatility calculation
            weights[i] = 1.0
            continue
            
        current_return = daily_returns.iloc[i]
        current_volatility = avg_range_20d.iloc[i]
        
        # Higher weight for negative return days with low volatility
        if current_return < 0 and current_volatility < avg_range_20d.iloc[i-10:i].median():
            weights[i] = 2.0
        # Lower weight for positive return days with high volatility
        elif current_return > 0 and current_volatility > avg_range_20d.iloc[i-10:i].median():
            weights[i] = 0.5
        else:
            weights[i] = 1.0
    
    # Apply weights to momentum
    factor = momentum_10d * weights
    
    return factor
