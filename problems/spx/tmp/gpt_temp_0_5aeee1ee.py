import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Adjusted Close-to-Open Return by Volume
    close_open_return = (df['close'] - df['open']) / df['open']
    close_open_return_adj = close_open_return * df['volume'] / df['volume'].rolling(window=5).sum()
    
    # Weighted Sum of High-Low Spread and Adjusted Close-to-Open Return
    avg_volume = df['volume'].rolling(window=5).mean()
    weighted_sum = (high_low_spread + close_open_return_adj) / avg_volume
    
    # Enhanced Price Momentum
    daily_log_return = np.log(df['close'] / df['close'].shift(1))
    price_momentum = daily_log_return.rolling(window=10).sum()
    
    # Volume Influence
    relative_volume_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    volume_influence = relative_volume_change.rolling(window=10).sum()
    
    # Combine Price and Volume
    price_volume_combined = price_momentum * volume_influence
    
    # Advanced Volume-Sensitivity Index (AVSI)
    close_minus_open = df['close'] - df['open']
    relative_volume_fluctuation = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    amount_fluctuation = (df['amount'] - df['amount'].shift(1)) / df['amount'].shift(1)
    avsi_factor = (close_minus_open * relative_volume_fluctuation) + (amount_fluctuation * 0.5)
    
    # Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Adjust by Open Price
    adjusted_high_low_spread = high_low_spread - df['open']
    
    # Combine Intraday Return and Adjusted High-Low Spread
    combined_intraday = intraday_return * adjusted_high_low_spread
    
    # Volume-Weighted Combination
    volume_weighted_combination = df['volume'] * combined_intraday
    
    # Inverse Volume Weighting
    inverse_volume_weighting = np.where(df['volume'] > df['volume'].rolling(window=5).mean(), 
                                        1 / (df['volume'] / df['volume'].rolling(window=5).mean()), 
                                        1)
    
    # Adjust Volume-Weighted Combination
    adjusted_volume_weighted = volume_weighted_combination * inverse_volume_weighting
    
    # Final Alpha Factor
    final_alpha = (weighted_sum + price_volume_combined + avsi_factor + adjusted_volume_weighted).fillna(0)
    
    # Filter Positive Values
    final_alpha = np.where(final_alpha > 0, final_alpha, 0)
    
    return final_alpha
