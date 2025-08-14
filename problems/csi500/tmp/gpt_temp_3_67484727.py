import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum
    price_momentum = df['close'] - df['close'].shift(10)
    
    # Calculate Volume Momentum
    avg_volume_5_days = df['volume'].rolling(window=5).mean().shift(1)
    volume_momentum = df['volume'] - avg_volume_5_days
    
    # Calculate 10-day Average True Range (ATR)
    true_range = df[['high', 'low']].diff(axis=1).abs()['high']
    true_range = true_range.combine(df['close'].shift(1), max, axis=0)
    atr = true_range.rolling(window=10).mean()
    
    # Combine Price, Volume, and Volatility
    combined_factor = (price_momentum * volume_momentum + atr + 1e-6) / df['volume']
    
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Adjust by Volume
    adjusted_high_low_spread = high_low_spread * df['volume']
    
    # Incorporate Lagged Close Price
    lagged_close = df['close'].shift(1)
    adjusted_spread_per_lagged_close = adjusted_high_low_spread / lagged_close
    
    # Add Momentum Component
    momentum = df['close'] - df['close'].shift(5)
    combined_with_momentum = adjusted_spread_per_lagged_close + momentum
    
    # Introduce Volatility Component
    open_log = np.log(df['open'])
    high_log = np.log(df['high'])
    low_log = np.log(df['low'])
    close_log = np.log(df['close'])
    daily_volatility = 0.5 * (high_log - low_log) ** 2 - (2 * np.log(2) - 1) * (close_log - open_log) ** 2
    final_combined_factor = combined_with_momentum * daily_volatility
    
    # Final Combination
    alpha_factor = (combined_factor + final_combined_factor) / 2
    
    return alpha_factor
