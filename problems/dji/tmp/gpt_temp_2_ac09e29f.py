import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Price
    df['volume_weighted_price'] = df['close'] * df['volume']
    
    # Calculate Daily Log Return
    df['daily_log_return'] = np.log(df['volume_weighted_price'] / df['volume_weighted_price'].shift(1))
    
    # Calculate 20-Day Moving Average of Close Price
    df['20_day_ma_close'] = df['close'].rolling(window=20).mean()
    
    # Calculate 20-Day Standard Deviation of Log Returns
    df['20_day_std_log_return'] = df['daily_log_return'].rolling(window=20).std()
    
    # Calculate Trend Momentum Indicator
    df['trend_momentum_indicator'] = (df['close'] - df['20_day_ma_close']) / df['20_day_std_log_return']
    
    # Calculate Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Smooth the Daily Log Returns using a Simple Moving Average
    df['smoothed_daily_log_return'] = df['daily_log_return'].rolling(window=10).mean()
    
    # Sum 5-Day Smoothed Returns
    df['sum_5_day_smoothed_returns'] = df['smoothed_daily_log_return'].rolling(window=5).sum()
    
    # Calculate Volume Trend
    df['volume_trend'] = df['volume'] - df['volume'].shift(m)  # m is the number of periods for volume trend
    
    # Adjust Momentum by Volume Trend
    df['adjusted_sum_5_day_smoothed_returns'] = df['sum_5_day_smoothed_returns'] * df['volume_trend']
    
    # Confirm Momentum with Volume Surge
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['confirmed_momentum'] = np.where(df['volume_change'] > 0, df['adjusted_sum_5_day_smoothed_returns'], 0)
    
    # Enhance the Factor
    df['enhanced_factor'] = df['confirmed_momentum'] * np.abs(df['volume_change'])
    df['enhanced_factor'] = np.where(df['enhanced_factor'] > 2.0, df['enhanced_factor'], 0)
    
    # Final Alpha Factor
    df['trend_intraday_component'] = (df['intraday_high_low_spread'] * df['close_to_open_return']) / df['20_day_std_log_return']
    df['final_alpha_factor'] = df['trend_intraday_component'] + df['enhanced_factor']
    
    return df['final_alpha_factor']
