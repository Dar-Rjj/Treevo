import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adaptive Momentum Divergence with Exponential Smoothing
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate daily range for price volatility
    daily_range = df['high'] - df['low']
    
    # Calculate daily volume change for volume volatility
    daily_vol_change = df['volume'].diff().abs()
    
    # Initialize EMA series
    ema_range = pd.Series(index=df.index, dtype=float)
    ema_vol_change = pd.Series(index=df.index, dtype=float)
    ema_volume = pd.Series(index=df.index, dtype=float)
    
    # Calculate EMAs with exponential smoothing
    alpha_range = 0.15
    alpha_vol_change = 0.15
    alpha_volume = 0.1
    
    # Initialize first values
    ema_range.iloc[0] = daily_range.iloc[0]
    ema_vol_change.iloc[0] = daily_vol_change.iloc[0] if not pd.isna(daily_vol_change.iloc[0]) else 0
    ema_volume.iloc[0] = df['volume'].iloc[0]
    
    # Calculate EMAs iteratively
    for i in range(1, len(df)):
        ema_range.iloc[i] = alpha_range * daily_range.iloc[i] + (1 - alpha_range) * ema_range.iloc[i-1]
        ema_vol_change.iloc[i] = alpha_vol_change * daily_vol_change.iloc[i] + (1 - alpha_vol_change) * ema_vol_change.iloc[i-1]
        ema_volume.iloc[i] = alpha_volume * df['volume'].iloc[i] + (1 - alpha_volume) * ema_volume.iloc[i-1]
    
    # Calculate price momentum components
    close = df['close']
    
    # Fast price momentum (5-day)
    fast_price_raw = (close / close.shift(5)) - 1
    fast_price_vol_adj = fast_price_raw / (ema_range + 1e-6)
    
    # Medium price momentum (10-day)
    medium_price_raw = (close / close.shift(10)) - 1
    medium_price_vol_adj = medium_price_raw / (ema_range + 1e-6)
    
    # Slow price momentum (20-day)
    slow_price_raw = (close / close.shift(20)) - 1
    slow_price_vol_adj = slow_price_raw / (ema_range + 1e-6)
    
    # Calculate volume momentum components
    volume = df['volume']
    
    # Fast volume momentum (5-day)
    fast_volume_ratio = volume / (ema_volume.shift(5) + 1e-6)
    fast_volume_vol_adj = fast_volume_ratio / (ema_vol_change + 1e-6)
    
    # Medium volume momentum (10-day)
    medium_volume_ratio = volume / (ema_volume.shift(10) + 1e-6)
    medium_volume_vol_adj = medium_volume_ratio / (ema_vol_change + 1e-6)
    
    # Slow volume momentum (20-day)
    slow_volume_ratio = volume / (ema_volume.shift(20) + 1e-6)
    slow_volume_vol_adj = slow_volume_ratio / (ema_vol_change + 1e-6)
    
    # Calculate momentum divergences
    fast_divergence = fast_price_vol_adj - fast_volume_vol_adj
    medium_divergence = medium_price_vol_adj - medium_volume_vol_adj
    slow_divergence = slow_price_vol_adj - slow_volume_vol_adj
    
    # Calculate volatility regime signal and dynamic weights
    volatility_ratio = ema_range / (ema_range.shift(10) + 1e-6)
    
    fast_weight = np.maximum(0, 2.0 - volatility_ratio)
    medium_weight = np.exp(-np.abs(1 - volatility_ratio))
    slow_weight = np.maximum(0, volatility_ratio - 1.0)
    
    # Calculate final alpha factor
    result = (fast_divergence * fast_weight + 
              medium_divergence * medium_weight + 
              slow_divergence * slow_weight)
    
    return result
