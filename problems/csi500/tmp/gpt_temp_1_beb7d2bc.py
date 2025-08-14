import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 7-Day and 28-Day EMA of Close Prices
    df['7_day_ema'] = df['close'].ewm(span=7, adjust=False).mean()
    df['28_day_ema'] = df['close'].ewm(span=28, adjust=False).mean()
    
    # Calculate Momentum Difference
    df['momentum_diff'] = df['7_day_ema'] - df['28_day_ema']
    
    # Calculate Weighted Returns
    df['5_day_return'] = df['close'].pct_change(5)
    df['10_day_return'] = df['close'].pct_change(10)
    df['20_day_return'] = df['close'].pct_change(20)
    df['weighted_returns'] = 0.4 * df['5_day_return'] + 0.3 * df['10_day_return'] + 0.3 * df['20_day_return']
    
    # Enhance Momentum Component
    df['enhanced_momentum'] = df['momentum_diff'] + df['weighted_returns']
    
    # Calculate High-Low Range and Close Position in Range
    df['high_low_range'] = df['high'] - df['low']
    df['close_position_in_range'] = (df['close'] - df['low']) / df['high_low_range']
    
    # Calculate Volume-Weighted Position
    df['volume_weighted_position'] = df['close_position_in_range'] * np.sqrt(df['volume'])
    
    # Calculate Moving Average of Volume-Weighted Position
    df['15_day_moving_avg_vol_weighted_pos'] = df['volume_weighted_position'].rolling(window=15).mean()
    
    # Combine Components
    df['combined_momentum'] = df['momentum_diff'] * df['15_day_moving_avg_vol_weighted_pos']
    
    # Integrate Volume Information
    df['5_day_avg_volume'] = df['volume'].rolling(window=5).mean()
    df['volume_adjusted_momentum'] = df['momentum_diff'] / (df['5_day_avg_volume'] + 1e-6)
    
    # Adjust by Amplitude of Price Movement
    df['7_day_price_range'] = df['high'].rolling(window=7).max() - df['low'].rolling(window=7).min()
    df['amplitude_adjusted_momentum'] = df['combined_momentum'] * df['7_day_price_range']
    
    # Final Adjustment
    df['non_linear_transform'] = df['amplitude_adjusted_momentum'].apply(lambda x: x ** (1/3))
    df['15_day_avg_close'] = df['close'].rolling(window=15).mean()
    df['final_factor'] = df['non_linear_transform'] - df['15_day_avg_close']
    
    # Incorporate Volatility
    df['14_day_atr'] = df[['high', 'low', 'close']].apply(lambda x: x.diff().abs().max(axis=1), axis=1).rolling(window=14).mean()
    df['normalized_momentum'] = df['momentum_diff'] / df['14_day_atr']
    
    # Incorporate Volume Trend
    df['10_day_ema_vol'] = df['volume'].ewm(span=10, adjust=False).mean()
    df['21_day_ema_vol'] = df['volume'].ewm(span=21, adjust=False).mean()
    df['final_factor'] = df.apply(lambda row: row['final_factor'] * 1.2 if row['10_day_ema_vol'] > row['21_day_ema_vol'] else row['final_factor'], axis=1)
    
    # Final Factor
    return df['final_factor']
