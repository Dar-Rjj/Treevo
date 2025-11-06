import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Short-term price acceleration (3-day momentum normalized by 1-day range)
    price_acceleration = (close / close.shift(3) - 1) / ((high - low) / close)
    
    # Long-term volatility-adjusted momentum (20-day return divided by 20-day volatility)
    returns_20d = close.pct_change(20)
    vol_20d = close.pct_change().rolling(20).std()
    vol_adjusted_momentum = returns_20d / vol_20d
    
    # Volume confirmation signal (volume trend vs price trend)
    volume_trend = volume.rolling(5).mean() / volume.rolling(20).mean()
    price_trend = close / close.shift(5)
    volume_confirmation = np.sign(volume_trend) * np.sign(price_trend - 1)
    
    # Combine components
    heuristics_matrix = price_acceleration * vol_adjusted_momentum * volume_confirmation
    
    return heuristics_matrix
