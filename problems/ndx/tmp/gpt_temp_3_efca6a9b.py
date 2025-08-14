import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=20):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Price Change
    df['price_change'] = df['close'] - df['close'].shift(1)
    
    # Detect Volume Spikes
    df['average_volume'] = df['volume'].rolling(window=N).mean()
    df['volume_spike'] = df['volume'] > 2 * df['average_volume']
    
    # Adjust Price Change by Intraday Range
    df['adjusted_price_change'] = df['price_change'] / df['intraday_range']
    
    # Weight by Volume
    df['weighted_adjusted_price_change'] = df['volume'] * df['adjusted_price_change']
    
    # Enhance Momentum on Volume Spike Days
    df['enhanced_weighted_adjusted_price_change'] = np.where(
        df['volume_spike'], 
        1.5 * df['weighted_adjusted_price_change'], 
        df['weighted_adjusted_price_change']
    )
    
    # Cumulative Momentum
    df['cumulative_momentum'] = df['enhanced_weighted_adjusted_price_change'].rolling(window=N).sum()
    
    # Calculate Average True Range (ATR)
    df['true_range'] = df[['high' - 'low', abs('high' - df['close'].shift(1)), abs('low' - df['close'].shift(1))]].max(axis=1)
    df['atr'] = df['true_range'].rolling(window=N).mean()
    
    # Adjust Cumulative Momentum by ATR
    df['adjusted_cumulative_momentum'] = df['cumulative_momentum'] / df['atr']
    
    # Calculate Relative Strength
    min_close = df['close'].rolling(window=N).min()
    max_close = df['close'].rolling(window=N).max()
    df['relative_strength'] = (df['close'] - min_close) / (max_close - min_close)
    
    # Combine Momentum and Relative Strength
    df['final_alpha_factor'] = df['adjusted_cumulative_momentum'] + df['relative_strength']
    
    return df['final_alpha_factor']
