import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Price Movement Ratio
    price_movement_ratio = (df['close'] / df['open']) - 1
    
    # Calculate 5-day Rolling Sum of Price Movement Ratio
    rolling_sum_price_movement = price_movement_ratio.rolling(window=5).sum()
    
    # Calculate 5-day Rolling Sum of High-Low Range
    rolling_sum_high_low_range = high_low_range.rolling(window=5).sum()
    
    # Compute Momentum Indicator
    momentum_indicator = rolling_sum_price_movement / rolling_sum_high_low_range
    
    # Adjust for Volume
    volume_adjusted_momentum = momentum_indicator * df['volume']
    
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Compute Volume Change Indicator
    volume_change_indicator = np.where(df['volume'].diff() > 0, 1, -1)
    
    # Combine Intraday and Volume-Amount Indicators
    combined_intraday_volume = intraday_return * volume_change_indicator
    
    # Apply Exponential Moving Average for Smoothing
    smoothing_factor = 0.2
    smoothed_combined = combined_intraday_volume.ewm(alpha=smoothing_factor, adjust=False).mean()
    
    # Calculate Daily Volatility
    daily_volatility = (df['high'] - df['low']).abs()
    
    # Combine Factors
    combined_factors = (intraday_return * df['volume']) / daily_volatility
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] / df['open']) - 1
    
    # Integrate Both Momentum Indicators
    integrated_momentum = close_to_open_return + rolling_sum_price_movement
    
    # Incorporate Volume and Amount Momentum
    avg_volume_9d = df['volume'].rolling(window=9).mean().shift(1)
    avg_amount_9d = df['amount'].rolling(window=9).mean().shift(1)
    
    volume_change = df['volume'] - avg_volume_9d
    amount_change = df['amount'] - avg_amount_9d
    
    combined_volume_amount_momentum = (volume_change + amount_change) * high_low_range
    
    # Apply Time Series Momentum
    alpha_factor_ma = combined_factors.rolling(window=10).mean()
    time_series_momentum = combined_factors - alpha_factor_ma
    
    # Final Alpha Factor
    final_alpha_factor = np.where(volume_change > 0, time_series_momentum, 0)
    
    return final_alpha_factor
