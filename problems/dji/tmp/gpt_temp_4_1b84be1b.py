import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Adjust Intraday Range by Volume
    volume_ema_span = 10
    volume_ema = df['volume'].ewm(span=volume_ema_span).mean()
    adjusted_volume = df['volume'] / volume_ema
    adjusted_intraday_range = intraday_range * adjusted_volume
    
    # Further Adjustment by Close Price Volatility
    true_range = np.maximum.reduce([df['high'] - df['low'], 
                                    abs(df['high'] - df['close'].shift(1)), 
                                    abs(df['low'] - df['close'].shift(1))])
    tr_std_dev_lookback = 20
    close_volatility = true_range.rolling(window=tr_std_dev_lookback).std()
    
    # Divide Adjusted Intraday Range by Close Price Volatility
    adjusted_intraday_range = adjusted_intraday_range / close_volatility
    
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Combine Adjusted Intraday Range and High-Low Spread
    combined_value = adjusted_intraday_range + high_low_spread
    
    # Weight by Volume
    volume_weighted_value = combined_value * df['volume']
    
    # Calculate 5-Day Average of Combined Value
    five_day_average = volume_weighted_value.rolling(window=5).mean()
    
    # Subtract 5-Day Average from Volume-Weighted Combined Value
    alpha_factor = volume_weighted_value - five_day_average
    
    # Further Adjustment by Open Price Volatility and Return Momentum
    open_log_returns = np.log(df['open']) - np.log(df['open'].shift(1))
    open_volatility = open_log_returns.rolling(window=tr_std_dev_lookback).std()
    
    close_log_returns = np.log(df['close']) - np.log(df['close'].shift(1))
    momentum = close_log_returns.rolling(window=10).mean()
    
    # Final Alpha Factor
    final_alpha_factor = (alpha_factor / open_volatility) - momentum
    
    # Exponential Moving Average
    ema_span = 14
    final_alpha_factor = final_alpha_factor.ewm(span=ema_span).mean()
    
    return final_alpha_factor.dropna()
