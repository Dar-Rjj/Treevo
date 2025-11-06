import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Short-term price acceleration (3-day ROC of 5-day ROC)
    roc_5 = close.pct_change(5)
    price_acceleration = roc_5.pct_change(3)
    
    # Long-term volatility-scaled momentum (20-day return divided by 20-day volatility)
 returns_20 = close.pct_change(20)
    volatility_20 = close.pct_change().rolling(20).std()
    vol_scaled_momentum = returns_20 / volatility_20
    
    # Volume trend strength (5-day vs 20-day volume ratio change)
    volume_ratio = volume.rolling(5).mean() / volume.rolling(20).mean()
    volume_trend = volume_ratio.pct_change(3)
    
    # Core factor: acceleration minus momentum, adjusted by volume trend
    raw_factor = price_acceleration - vol_scaled_momentum
    heuristics_matrix = raw_factor * (1 + volume_trend)
    
    return heuristics_matrix
