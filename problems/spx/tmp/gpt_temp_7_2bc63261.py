import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Calculate High-Low Spread and Adjust by Open Price
    high_low_spread = df['high'] - df['low']
    adjusted_high_low_spread = high_low_spread / df['open']
    
    # Combine Intraday Return and Adjusted High-Low Spread
    combined_factor = intraday_return * adjusted_high_low_spread
    
    # Volume Weighting
    volume_weighted_combination = combined_factor * df['volume']
    
    # Detect Volume Spike
    average_volume_5d = df['volume'].rolling(window=5).mean()
    volume_spike = (df['volume'] > average_volume_5d)
    
    # Apply Inverse Volume Weighting
    inverse_volume_weight = 1 / (df['volume'] / average_volume_5d) if volume_spike else 1
    adjusted_volume_weighted = volume_weighted_combination * inverse_volume_weight
    
    # Calculate Daily Momentum
    daily_momentum = df['close'] - df['close'].shift(1)
    
    # Combine Daily Momentum and Adjusted Volume-Weighted Combination
    combined_momentum_volume = daily_momentum * adjusted_volume_weighted
    
    # Filter Positive Values
    positive_values = combined_momentum_volume.where(combined_momentum_volume > 0, 0)
    
    # Final Alpha Factor: Sum of Positive Values over a rolling window
    final_alpha_factor = positive_values.rolling(window=21).sum()
    
    # Incorporate Price Trend
    price_trend_21d = df['close'] - df['close'].rolling(window=21).mean()
    final_alpha_factor_with_trend = final_alpha_factor + price_trend_21d
    
    return final_alpha_factor_with_trend
