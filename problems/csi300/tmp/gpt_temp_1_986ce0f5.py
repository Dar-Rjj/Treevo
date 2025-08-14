import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Gaps
    df['daily_gap'] = df['open'] - df['close'].shift(1)
    
    # Calculate Volume Weighted Average of Gaps
    df['volume_weighted_gap'] = (df['daily_gap'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Incorporate Enhanced Price Momentum
    short_ema_window = 10
    long_ema_window = 30
    
    df['short_ema'] = df['close'].ewm(span=short_ema_window, adjust=False).mean()
    df['long_ema'] = df['close'].ewm(span=long_ema_window, adjust=False).mean()
    df['change_in_ema'] = df['short_ema'] - df['short_ema'].shift(1)
    
    # Introduce Dynamic ATR Component
    df['true_range'] = df[['high' - 'low', 'high' - df['close'].shift(1), 'low' - df['close'].shift(1)]].max(axis=1)
    df['aatr'] = df['true_range'].ewm(span=14, adjust=False).mean()
    df['adjusted_momentum'] = df['change_in_ema'] / df['aatr']
    
    # Combine Factors and Adjust for Volume
    volume_ewma_window = 5
    df['volume_ewma'] = df['volume'].ewm(span=volume_ewma_window, adjust=False).mean()
    df['volume_adjusted_momentum'] = df['adjusted_momentum'] * (df['volume'] / df['volume_ewma'])
    
    # Price Reversal Sensitivity with Volume Influence
    df['high_low_spread'] = df['high'] - df['low']
    df['weighted_high_low_spread'] = df['high_low_spread'] * df['volume']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['volume_weighted_gap'] + df['volume_adjusted_momentum'] - df['weighted_high_low_spread']
    
    return df['alpha_factor']
