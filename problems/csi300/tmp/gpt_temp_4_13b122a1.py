import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate Exponential Moving Average (EMA) of High-Low Range
    lookback_period = 20
    df['ema_high_low_range'] = df['high_low_range'].ewm(span=lookback_period, adjust=False).mean()
    
    # Calculate Volume Adjusted High-Low Range
    df['volume_adjusted_high_low_range'] = df['high_low_range'] * df['volume']
    
    # Calculate EMA of Volume Adjusted High-Low Range
    df['ema_volume_adjusted_high_low_range'] = df['volume_adjusted_high_low_range'].ewm(span=lookback_period, adjust=False).mean()
    
    # Calculate Momentum Factor
    df['momentum_factor'] = df['ema_volume_adjusted_high_low_range'] - df['ema_high_low_range']
    
    # Integrate Volatility
    df['daily_return'] = df['close'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=lookback_period).std()
    df['volatility_adjusted_momentum'] = df['momentum_factor'] / df['volatility']
    
    # Incorporate Relative Strength
    period = 60
    df['stock_return'] = df['close'].pct_change(period)
    df['index_return'] = df['index_close'].pct_change(period)
    df['relative_strength'] = df['stock_return'] - df['index_return']
    
    # Incorporate Sector Momentum
    df['sector_index_return'] = df['sector_index_close'].pct_change(period)
    df['sector_momentum'] = df['stock_return'] - df['sector_index_return']
    
    # Incorporate Volume Spikes
    threshold = 2.0
    df['volume_spike'] = (df['volume'] / df['volume'].shift(1)) > threshold
    df['volume_spike'] = df['volume_spike'].astype(int)
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['volatility_adjusted_momentum'] + df['relative_strength'] + df['sector_momentum']) * df['volume_spike']
    
    return df['alpha_factor']
