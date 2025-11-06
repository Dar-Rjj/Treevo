import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Calculate absolute returns
    abs_returns = returns.abs()
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate 20-day volume moving average
    volume_ma = df['volume'].rolling(window=20, min_periods=10).mean()
    
    # Calculate volume trend (slope of 20-day volume MA)
    volume_trend = volume_ma.diff(5)  # 5-day change in 20-day MA as slope proxy
    
    # Calculate correlations for each day using rolling windows
    for i in range(20, len(df)):
        # Get rolling window of past 60 days
        start_idx = max(0, i - 59)
        window_returns = returns.iloc[start_idx:i+1]
        window_abs_returns = abs_returns.iloc[start_idx:i+1]
        window_volume = df['volume'].iloc[start_idx:i+1]
        
        # Filter positive return days
        positive_mask = window_returns > 0
        if positive_mask.sum() >= 5:  # Minimum 5 observations
            pos_corr = window_volume[positive_mask].corr(window_abs_returns[positive_mask])
        else:
            pos_corr = 0
        
        # Filter negative return days
        negative_mask = window_returns < 0
        if negative_mask.sum() >= 5:  # Minimum 5 observations
            neg_corr = window_volume[negative_mask].corr(window_abs_returns[negative_mask])
        else:
            neg_corr = 0
        
        # Calculate asymmetry
        if pd.notna(pos_corr) and pd.notna(neg_corr):
            asymmetry = pos_corr - neg_corr
        else:
            asymmetry = 0
        
        # Multiply by volume trend (handle NaN)
        current_volume_trend = volume_trend.iloc[i] if pd.notna(volume_trend.iloc[i]) else 0
        
        factor.iloc[i] = asymmetry * current_volume_trend
    
    # Fill initial NaN values with 0
    factor = factor.fillna(0)
    
    return factor
