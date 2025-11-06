import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Price oscillation extreme detection
    median_price = (high + low) / 2
    price_oscillation = (close - median_price) / (high - low + 1e-8)
    oscillation_extreme = price_oscillation.rolling(8).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
    
    # Volume-confirmed breakout strength
    volume_surge = volume / volume.rolling(15).mean()
    breakout_range = (high.rolling(5).max() - low.rolling(5).min()) / (high.rolling(15).max() - low.rolling(15).min() + 1e-8)
    volume_breakout = volume_surge * breakout_range
    
    # Adaptive market noise filtration
    noise_ratio = (close.diff().abs() / (high - low + 1e-8)).rolling(10).mean()
    trend_purity = 1 - noise_ratio
    
    # Volatility-regime adjustment
    vol_regime = (high - low).rolling(10).std() / (high - low).rolling(30).std()
    
    # Composite reversal factor
    heuristics_matrix = oscillation_extreme * volume_breakout * trend_purity * vol_regime
    
    return heuristics_matrix
