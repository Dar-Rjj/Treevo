import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Price volatility normalized by price level
    price_range = (high - low) / ((high + low) / 2)
    
    # Relative volume strength compared to 5-day average
    volume_ratio = volume / volume.rolling(window=5).mean()
    
    # Momentum acceleration (rate of change of momentum)
    momentum = close / close.shift(5) - 1
    momentum_accel = momentum - momentum.shift(3)
    
    # Mean-reversion component based on distance from recent high/low
    mid_price = (high + low) / 2
    recent_high = high.rolling(window=10).max()
    recent_low = low.rolling(window=10).min()
    price_position = (mid_price - recent_low) / (recent_high - recent_low)
    mean_reversion = 0.5 - price_position
    
    # Combine components with emphasis on volatility and mean-reversion
    heuristics_matrix = price_range * volume_ratio * momentum_accel * mean_reversion
    
    return heuristics_matrix
