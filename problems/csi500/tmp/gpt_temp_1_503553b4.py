import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Enhanced High-to-Low Price Ratio
    high_low_ratio = df['high'] / df['low']
    
    # Evaluate Volume Trend
    volume_10_ma = df['volume'].rolling(window=10).mean()
    volume_trend = np.where(df['volume'] > volume_10_ma, 1, -1)
    
    # Integrate Components for Adjusted High-to-Low Price Ratio
    adjusted_high_low_ratio = high_low_ratio * volume_trend
    
    # Calculate 30-day Price Momentum
    price_momentum_30 = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    
    # Identify Volume Shock
    volume_30_ma = df['volume'].rolling(window=30).mean().shift(1)  # Exclude day t
    volume_shock = np.where(df['volume'] > 2 * volume_30_ma, 1, 0)
    
    # Multiply 30-day Momentum by Volume Shock indicator
    momentum_with_volume_shock = price_momentum_30 * volume_shock
    
    # Calculate Adjusted High-to-Low Range
    high_low_range = df['high'] - df['low']
    adjusted_high_low_range = high_low_range * np.sqrt(df['volume'])
    
    # Detect Volume Spike
    volume_5_ma = df['volume'].rolling(window=5).mean()
    volume_spike = np.where(df['volume'] > 1.7 * volume_5_ma, 1, 0)
    
    # Calculate 10-day Exponential Moving Average (EMA) of Close price
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    
    # Combine Adjusted High-to-Low Range, 10-day Momentum, and Volume Spike
    combined_factor = adjusted_high_low_range + price_momentum_30
    combined_factor = combined_factor * (2.5 if volume_spike == 1 else 1)
    
    # Synthesize Final Alpha Factor
    final_alpha_factor = (momentum_with_volume_shock + combined_factor) - ema_10
    
    return final_alpha_factor
