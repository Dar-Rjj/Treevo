import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Volume acceleration with directional confirmation
    volume_acceleration = volume / volume.rolling(window=10).mean()
    directional_efficiency = (close - close.shift(5)) / (high.rolling(window=5).max() - low.rolling(window=5).min())
    
    # Price-volume divergence detection
    price_trend = close.rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_trend = volume.rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    divergence = np.sign(price_trend) != np.sign(volume_trend)
    
    # Combine factors with divergence as signal amplifier
    heuristics_matrix = volume_acceleration * directional_efficiency * np.where(divergence, -1, 1)
    
    return heuristics_matrix
