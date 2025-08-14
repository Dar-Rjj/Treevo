import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Adjust Intraday Range by Volume
    df['volume_ema'] = df['volume'].ewm(span=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ema']
    df['adjusted_intraday_range'] = df['intraday_range'] * df['volume_ratio']
    
    # Further Adjustment by Close Price Volatility
    df['close_returns'] = df['close'].pct_change()
    df['close_volatility'] = df['close_returns'].rolling(window=10).std()
    df['adjusted_intraday_range_by_volatility'] = df['adjusted_intraday_range'] / df['close_volatility']
    
    # Calculate True Range Using High, Low, and Close Prices
    df['true_range'] = df[['high', 'low']].diff(axis=1).iloc[:, 0].abs()
    df['true_range'] = df.apply(lambda x: max(x['high'] - x['low'], 
                                              abs(x['high'] - x['close'].shift(1)), 
                                              abs(x['low'] - x['close'].shift(1))), axis=1)
    
    # Adjust Intraday Range by True Range Volatility
    df['true_range_volatility'] = df['true_range'].rolling(window=10).std()
    df['adjusted_intraday_range_by_true_range_volatility'] = df['adjusted_intraday_range'] / df['true_range_volatility']
    
    # Calculate Daily High-Low Difference
    df['high_low_diff'] = df['high'] - df['low']
    
    # Compute Exponential Moving Average (EMA) of High-Low Difference
    df['high_low_diff_ema'] = df['high_low_diff'].ewm(span=5).mean()
    
    # Compute High-Low Momentum
    df['high_low_momentum'] = df['high_low_diff'] - df['high_low_diff_ema'].shift(1)
    
    # Calculate Price Momentum
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['price_momentum'] = df['ma_10'] - df['ma_20']
    
    # Enhance Price-Velocity Factors
    df['roc_5'] = df['close'].pct_change(periods=5)
    df['roc_10'] = df['close'].pct_change(periods=10)
    df['roc_volume_3'] = df['volume'].pct_change(periods=3)
    df['roc_volume_7'] = df['volume'].pct_change(periods=7)
    
    # Synthesize Intraday, High-Low, and Price-Volume Momentum
    df['intraday_momentum'] = df['adjusted_intraday_range_by_volatility']
    df['intraday_true_range_momentum'] = df['adjusted_intraday_range_by_true_range_volatility']
    df['high_low_momentum_vol_ratio'] = df['high_low_momentum'] * df['volume_ratio']
    df['price_momentum_vol_ratio'] = df['price_momentum'] * df['volume_ratio']
    df['price_velocity_vol_ratio'] = (df['roc_5'] + df['roc_10']) * df['volume_ratio']
    df['volume_velocity_vol_ratio'] = (df['roc_volume_3'] + df['roc_volume_7']) * df['volume_ratio']
    
    df['factor'] = (df['intraday_momentum'] + 
                    df['intraday_true_range_momentum'] + 
                    df['high_low_momentum_vol_ratio'] + 
                    df['price_momentum_vol_ratio'] + 
                    df['price_velocity_vol_ratio'] + 
                    df['volume_velocity_vol_ratio'])
    
    return df['factor']
