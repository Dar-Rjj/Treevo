import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Adjusted Close-to-Open Return by Volume
    close_to_open_return = (df['close'] / df['open']) - 1
    adjusted_close_to_open_return = (close_to_open_return * df['volume']) / df['volume'].rolling(window=5).sum()
    
    # Sum of High-Low Spread and Adjusted Close-to-Open Return
    sum_high_low_and_adjusted_return = high_low_spread + adjusted_close_to_open_return
    
    # Weight by Average Volume
    average_volume = df['volume'].rolling(window=5).mean()
    weighted_sum = sum_high_low_and_adjusted_return * (1 / average_volume)
    
    # Enhanced Price Momentum
    daily_log_return = np.log(df['close'] / df['close'].shift(1))
    price_momentum = daily_log_return.rolling(window=10).sum()
    
    # Incorporate Volume Influence
    volume_relative_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    volume_influence = volume_relative_change.rolling(window=10).sum()
    combined_price_volume = price_momentum * volume_influence
    
    # Advanced Volume-Sensitivity Index (AVSI)
    close_minus_open = df['close'] - df['open']
    relative_volume_fluctuation = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    avsi_factor = close_minus_open * relative_volume_fluctuation
    
    scaled_amount_fluctuation = (df['amount'] - df['amount'].shift(1)) / df['amount'].shift(1) * 0.5
    aggregate_avsi_factor = avsi_factor + scaled_amount_fluctuation
    
    # Calculate Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Adjust by Open Price
    adjusted_high_low_spread = high_low_spread - df['open']
    
    # Combine Intraday Return and Adjusted High-Low Spread
    combined_intraday_return_and_high_low = intraday_return * adjusted_high_low_spread
    
    # Volume Weighting
    volume_weighted_combination = df['volume'] * combined_intraday_return_and_high_low
    
    # Detect Volume Spike
    five_day_average_volume = df['volume'].rolling(window=5).mean()
    inverse_volume_weighting = 1 / (df['volume'] / five_day_average_volume)
    inverse_volume_weighting = np.where(df['volume'] > five_day_average_volume, inverse_volume_weighting, 1)
    
    # Adjust Volume-Weighted Combination by Inverse Volume Weighting
    adjusted_volume_weighted_combination = volume_weighted_combination * inverse_volume_weighting
    
    # Final Alpha Factor
    final_alpha_factor = (weighted_sum + combined_price_volume + 
                          aggregate_avsi_factor + adjusted_volume_weighted_combination)
    
    # Filter Positive Values
    final_alpha_factor = np.where(final_alpha_factor > 0, final_alpha_factor, 0)
    
    return final_alpha_factor
