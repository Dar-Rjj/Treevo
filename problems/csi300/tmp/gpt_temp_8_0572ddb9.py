import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Short-term momentum acceleration (3-day vs 6-day ROC difference)
    roc_3 = close.pct_change(3)
    roc_6 = close.pct_change(6)
    momentum_accel = roc_3 - roc_6
    
    # Medium-term mean reversion (10-day price position relative to range)
    high_10 = high.rolling(window=10).max()
    low_10 = low.rolling(window=10).min()
    range_position = (close - low_10) / (high_10 - low_10) - 0.5
    
    # Volatility regime weighting (20-day volume-adjusted volatility)
    returns = close.pct_change()
    vol_price = returns.rolling(window=20).std()
    vol_volume = volume.rolling(window=20).std()
    vol_regime = vol_price * vol_volume
    
    # Combine signals with volatility weighting
    combined_signal = momentum_accel * 0.6 + range_position * 0.4
    volatility_adjusted = combined_signal / vol_regime
    
    heuristics_matrix = volatility_adjusted
    
    return heuristics_matrix
