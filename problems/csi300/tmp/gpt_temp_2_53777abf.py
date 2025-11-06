import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Volatility-adjusted price momentum
    volatility = high.rolling(window=20).std()
    price_change = close.pct_change(periods=5)
    vol_adjusted_momentum = price_change / (volatility + 1e-8)
    
    # Volume-confirmed reversal detection
    volume_surge = volume / volume.rolling(window=30).mean()
    price_reversal = -close.pct_change(periods=3) * volume_surge
    
    # Combine factors with dynamic weighting
    combined_factor = vol_adjusted_momentum.rolling(window=10).mean() + price_reversal.rolling(window=5).mean()
    
    heuristics_matrix = combined_factor
    return heuristics_matrix
