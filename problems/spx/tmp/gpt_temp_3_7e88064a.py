import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Adjusted Close-to-Open Return by Volume
    close_open_return = (df['close'] - df['open']) / df['open']
    volume_sum_5d = df['volume'].rolling(window=5).sum()
    adjusted_close_open_return = close_open_return * df['volume'] / volume_sum_5d
    
    # Weighted Sum of High-Low Spread and Adjusted Close-to-Open Return
    avg_volume = volume_sum_5d / 5
    weighted_sum = high_low_spread + (adjusted_close_open_return / avg_volume)
    
    # Enhanced Price Momentum
    daily_log_return = np.log(df['close'] / df['close'].shift(1))
    enhanced_price_momentum = daily_log_return.rolling(window=10).sum()
    
    # Volume Influence
    relative_volume_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    summed_volume_change = relative_volume_change.rolling(window=10).sum()
    
    # Combine Price and Volume
    combined_price_volume = enhanced_price_momentum * summed_volume_change
    
    # Advanced Volume-Sensitivity Index (AVSI)
    close_minus_open = df['close'] - df['open']
    relative_volume_fluctuation = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    scaled_amount_fluctuation = (df['amount'] - df['amount'].shift(1)) / df['amount'].shift(1) * 0.5
    avsi_factor = close_minus_open * relative_volume_fluctuation + scaled_amount_fluctuation
    
    # Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Adjust by Open Price
    adjusted_high_low_spread = high_low_spread - df['open']
    
    # Combine Intraday Return and Adjusted High-Low Spread
    combined_intraday_adjusted_high_low = intraday_return * adjusted_high_low_spread
    
    # Volume-Weighted Combination
    volume_weighted_combination = df['volume'] * combined_intraday_adjusted_high_low
    
    # Dynamic Volume Weighting
    dynamic_volume_weighting = np.where(df['volume'] > volume_sum_5d / 5, 
                                        1 / (df['volume'] / (volume_sum_5d / 5)), 
                                        1)
    
    # Adjust Volume-Weighted Combination
    adjusted_volume_weighted_combination = volume_weighted_combination * dynamic_volume_weighting
    
    # Final Alpha Factor
    final_alpha_factor = (weighted_sum + combined_price_volume + 
                          avsi_factor + adjusted_volume_weighted_combination)
    
    # Filter Positive Values
    final_alpha_factor = np.where(final_alpha_factor > 0, final_alpha_factor, 0)
    
    return final_alpha_factor
