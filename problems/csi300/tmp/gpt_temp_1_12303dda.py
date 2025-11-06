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
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Calculate 20-day volume moving average
    volume_ma = df['volume'].rolling(window=20, min_periods=10).mean()
    
    # Calculate volume trend (slope)
    volume_trend = (volume_ma / volume_ma.shift(5) - 1).fillna(0)
    
    # Calculate correlations for each day using rolling window
    for i in range(len(df)):
        if i < 40:  # Need sufficient data for correlation calculation
            factor_values.iloc[i] = 0
            continue
            
        # Get data up to current day
        current_data = df.iloc[:i+1]
        current_returns = returns.iloc[:i+1]
        current_abs_returns = abs_returns.iloc[:i+1]
        
        # Identify positive and negative return days
        positive_mask = current_returns > 0
        negative_mask = current_returns < 0
        
        # Use last 30 days of data for correlation calculation
        recent_data = current_data.iloc[-30:]
        recent_returns = current_returns.iloc[-30:]
        recent_abs_returns = current_abs_returns.iloc[-30:]
        recent_positive_mask = positive_mask.iloc[-30:]
        recent_negative_mask = negative_mask.iloc[-30:]
        
        # Calculate positive return volume correlation
        if recent_positive_mask.sum() >= 5:  # Minimum 5 positive return days
            positive_volume = recent_data['volume'][recent_positive_mask]
            positive_abs_returns = recent_abs_returns[recent_positive_mask]
            pos_corr = positive_volume.corr(positive_abs_returns)
        else:
            pos_corr = 0
        
        # Calculate negative return volume correlation
        if recent_negative_mask.sum() >= 5:  # Minimum 5 negative return days
            negative_volume = recent_data['volume'][recent_negative_mask]
            negative_abs_returns = recent_abs_returns[recent_negative_mask]
            neg_corr = negative_volume.corr(negative_abs_returns)
        else:
            neg_corr = 0
        
        # Calculate correlation asymmetry
        correlation_asymmetry = pos_corr - neg_corr
        
        # Get current volume trend
        current_volume_trend = volume_trend.iloc[i] if not pd.isna(volume_trend.iloc[i]) else 0
        
        # Final factor value
        factor_values.iloc[i] = correlation_asymmetry * (1 + current_volume_trend)
    
    return factor_values.fillna(0)
