import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Adjust for Volume Change
    volume_change = df['volume'] - df['volume'].shift(1)
    high_low_adjusted = high_low_spread * np.where(volume_change > 0, volume_change, 0)
    
    # Apply Simple Moving Average (SMA) with a period of 5 days
    sma_5_high_low = high_low_adjusted.rolling(window=5).mean()
    
    # Calculate 10-Day Price Momentum
    price_momentum_10 = df['close'] - df['close'].shift(10)
    
    # Adjust for Volume Trend
    volume_trend = df['volume'] - df['volume'].rolling(window=10).mean()
    
    # Combine Price Momentum and Volume Trend Adjustment
    combined_momentum = price_momentum_10 * volume_trend
    combined_momentum = np.where(combined_momentum < 0, 0, combined_momentum)
    
    # Calculate Daily Log Return
    daily_log_return = np.log(df['close'] / df['close'].shift(1))
    
    # Identify Positive and Negative Returns
    positive_returns = daily_log_return > 0
    negative_returns = daily_log_return <= 0
    
    # Calculate Sum of Upward Volume and Downward Volume
    upward_volume = df['volume'][positive_returns].sum()
    downward_volume = df['volume'][negative_returns].sum()
    
    # Compute Trend Reversal Signal
    total_volume = upward_volume + downward_volume
    upward_volume_ratio = upward_volume / total_volume
    downward_volume_ratio = downward_volume / total_volume
    reversal_signal = np.where(upward_volume_ratio > downward_volume_ratio, 1, 0) * daily_log_return
    
    # Calculate Adjusted Daily Return
    gap = df['open'] - df['close'].shift(1)
    adjusted_daily_return = (df['close'] - df['close'].shift(1)) - np.where(gap > 0, gap, -gap)
    
    # Calculate Price Range Momentum
    current_range = df['high'] - df['low']
    previous_range = df['high'].shift(1) - df['low'].shift(1)
    price_range_momentum = current_range - previous_range
    
    # Volume Change Factor
    volume_change_factor = df['volume'] / df['volume'].shift(1)
    
    # Final Volume Weighted Momentum
    final_volume_weighted_momentum = combined_momentum * adjusted_daily_return
    
    # Integrate Momentum, Volume, and Price Change
    integrated_momentum = combined_momentum * (df['volume'] - df['volume'].shift(1)) + (df['close'] - df['close'].shift(1))
    
    # Final Alpha Factor
    alpha_factor = (combined_momentum + reversal_signal + final_volume_weighted_momentum +
                    adjusted_daily_return * price_range_momentum * volume_change_factor)
    alpha_factor = np.where(alpha_factor < 0, 0, alpha_factor)
    
    return alpha_factor
