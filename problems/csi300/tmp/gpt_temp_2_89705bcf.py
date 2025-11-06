import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence with Acceleration Momentum factor
    Combines acceleration momentum with volume-price divergence and regime detection
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Calculate Acceleration Momentum
    # First derivative (velocity)
    velocity = data['close'].diff()
    # Second derivative (acceleration) with asymmetric weighting
    acceleration = velocity.diff()
    # Asymmetric weighting: stronger weight for positive acceleration
    acceleration_weighted = np.where(acceleration > 0, acceleration * 1.2, acceleration * 0.8)
    
    # 2. Measure Volume-Price Divergence
    # Volume changes
    volume_changes = data['volume'].pct_change()
    # Price changes
    price_changes = data['close'].pct_change()
    
    # Rolling correlation between volume and price changes (20-day window)
    rolling_corr = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        window_volume = volume_changes.iloc[i-19:i+1]
        window_price = price_changes.iloc[i-19:i+1]
        if len(window_volume.dropna()) >= 10 and len(window_price.dropna()) >= 10:
            rolling_corr.iloc[i] = window_volume.corr(window_price)
        else:
            rolling_corr.iloc[i] = 0
    
    # Divergence strength (absolute deviation from normal correlation)
    divergence_strength = np.abs(rolling_corr - rolling_corr.rolling(60, min_periods=20).mean())
    
    # 3. Regime Detection using High-Low range
    # Daily range
    daily_range = (data['high'] - data['low']) / data['close']
    # Rolling range average
    range_ma_short = daily_range.rolling(10, min_periods=5).mean()
    range_ma_long = daily_range.rolling(50, min_periods=20).mean()
    
    # Regime classification: trending vs mean-reverting
    # High range persistence indicates trending, low range indicates mean-reverting
    range_ratio = range_ma_short / range_ma_long
    regime = np.where(range_ratio > 1.1, 'trending', 
                     np.where(range_ratio < 0.9, 'mean_reverting', 'neutral'))
    
    # 4. Adaptive Signal Combination
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(20, len(data)):
        current_acceleration = acceleration_weighted[i]
        current_divergence = divergence_strength.iloc[i]
        current_regime = regime[i]
        
        if current_regime == 'trending':
            # In trending regimes, weight acceleration by divergence strength
            factor_value = current_acceleration * (1 + current_divergence)
        elif current_regime == 'mean_reverting':
            # In mean-reverting regimes, invert the signal
            factor_value = -current_acceleration * (1 + current_divergence)
        else:
            # Neutral regime - use raw acceleration
            factor_value = current_acceleration
        
        factor_values.iloc[i] = factor_value
    
    # Normalize the factor
    factor_values = (factor_values - factor_values.rolling(60, min_periods=20).mean()) / \
                    factor_values.rolling(60, min_periods=20).std()
    
    return factor_values.fillna(0)
