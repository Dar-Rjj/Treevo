import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    N = 20  # Number of days for the rolling window
    short_term = 5
    long_term = 20
    
    # Calculate Volume-Weighted Average Prices
    df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['volume_weighted_price'] = df['avg_price'] * df['volume']
    df['sum_volume_weighted_price'] = df['volume_weighted_price'].rolling(window=N).sum()
    df['total_volume'] = df['volume'].rolling(window=N).sum()
    
    # Calculate Volume-Weighted Moving Average
    df['vwma'] = df['sum_volume_weighted_price'] / df['total_volume']
    
    # Calculate Current Day's Volume-Weighted Price
    df['current_day_vw_price'] = df['avg_price'] * df['volume']
    
    # Calculate VWPTI
    df['vwpti'] = (df['current_day_vw_price'] - df['vwma']) / df['vwma']
    
    # Calculate Short-Term and Long-Term Price Momentum
    df['short_term_momentum'] = df['close'] - df['close'].shift(short_term)
    df['long_term_momentum'] = df['close'] - df['close'].shift(long_term)
    
    # Calculate Exponential Moving Average (EMA) of Price Momentum
    alpha_short = 2 / (1 + short_term)
    alpha_long = 2 / (1 + long_term)
    
    df['short_term_ema'] = df['short_term_momentum'].ewm(alpha=alpha_short, adjust=False).mean()
    df['long_term_ema'] = df['long_term_momentum'].ewm(alpha=alpha_long, adjust=False).mean()
    
    # Calculate Volume Trend
    df['volume_trend'] = df['volume'].ewm(span=short_term, adjust=False).mean()
    
    # Incorporate Volume Adjusted Inertia
    df['positive_volume'] = np.where(df['close'] > df['close'].shift(1), df['volume'], 0)
    df['cumulative_volume_flow'] = df['positive_volume'].rolling(window=N).sum()
    
    # Weight by Volume
    df['up_down_count'] = (np.where(df['close'] > df['close'].shift(1), 1, -1)).rolling(window=N).sum()
    df['volume_weighted_directional_count'] = df['up_down_count'] * df['volume']
    
    # Combine Components
    df['combined_momentum'] = df['vwpti'] + df['volume_weighted_directional_count'] + df['short_term_momentum'] + df['long_term_momentum']
    df['final_alpha_factor'] = df['combined_momentum'] * df['short_term_ema'] * df['cumulative_volume_flow']
    
    return df['final_alpha_factor'].dropna()
