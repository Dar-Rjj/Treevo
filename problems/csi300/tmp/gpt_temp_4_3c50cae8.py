import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic price changes and returns
    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Momentum Efficiency Analysis
    intraday_efficiency = np.abs(close - open_price) / (high - low)
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    price_ratio = close / close.shift(1) - 1
    momentum_absorption = np.abs(close - open_price) / np.abs(price_ratio)
    momentum_absorption = momentum_absorption.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Efficiency Persistence
    efficiency_high_count = intraday_efficiency.rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x > 0.6), raw=True
    )
    efficiency_low_count = intraday_efficiency.rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x < 0.4), raw=True
    )
    efficiency_persistence = efficiency_high_count - efficiency_low_count
    
    # Volume-Price Alignment
    price_direction = np.sign(close - close.shift(1))
    directional_volume = price_direction * volume
    
    # Volume-Return Correlation
    volume_return_corr = directional_volume.rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x * np.abs(price_ratio.iloc[-len(x):])), raw=False
    )
    
    # Regime-Adaptive Momentum
    returns = close / close.shift(1) - 1
    vol_adjusted_momentum = returns / returns.rolling(window=10, min_periods=1).std()
    vol_adjusted_momentum = vol_adjusted_momentum.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    momentum_acceleration = (close / close.shift(2) - 1) - (close / close.shift(5) - 1)
    
    # Volume Confirmation
    volume_surge = volume / volume.rolling(window=5, min_periods=1).mean()
    volume_surge = volume_surge.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    volume_consistency = 1 / (volume.rolling(window=10, min_periods=1).std() / 
                             volume.rolling(window=10, min_periods=1).mean())
    volume_consistency = volume_consistency.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Alpha Construction
    efficiency_momentum = intraday_efficiency * momentum_acceleration
    aligned_momentum = volume_return_corr * vol_adjusted_momentum
    
    final_alpha = (efficiency_momentum + aligned_momentum) * volume_surge * volume_consistency
    
    return final_alpha
