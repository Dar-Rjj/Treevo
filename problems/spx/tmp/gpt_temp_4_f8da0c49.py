import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Adjust by Open Price
    adjusted_high_low_spread = high_low_spread - df['open']
    
    # Combine Intraday Return and Adjusted High-Low Spread
    combined_intraday_adjusted = intraday_return * adjusted_high_low_spread
    
    # Volume Weighting
    volume_weighted = combined_intraday_adjusted * df['volume']
    
    # Detect Volume Spike
    avg_volume_5d = df['volume'].rolling(window=5).mean()
    volume_spike = (df['volume'] > avg_volume_5d)
    
    # Apply Inverse Volume Weighting
    inverse_volume_weighting = volume_spike.apply(lambda x: 1 / (df['volume'] / avg_volume_5d) if x else 1)
    adjusted_volume_weighted = volume_weighted * inverse_volume_weighting
    
    # Calculate Daily Momentum
    daily_momentum = df['close'].diff(1)
    
    # Combine Daily Momentum and Adjusted Volume-Weighted Combination
    combined_momentum_volume = daily_momentum * adjusted_volume_weighted
    
    # Filter Positive Values
    combined_momentum_volume = combined_momentum_volume.where(combined_momentum_volume > 0, 0)
    
    # Final Alpha Factor
    final_alpha_factor = combined_momentum_volume.rolling(window=21).sum()
    
    # Calculate Volume-Weighted Close-to-Open Return
    close_to_open_return = df['close'] - df['open']
    total_volume_5d = df['volume'].rolling(window=5).sum()
    volume_weighted_c2o_return = (close_to_open_return * df['volume']) / total_volume_5d
    
    # Combine and Weight
    combined_hl_c2o = adjusted_high_low_spread + volume_weighted_c2o_return
    avg_volume_10d = df['volume'].rolling(window=10).mean()
    weighted_combined = combined_hl_c2o * (1 / avg_volume_10d)
    
    # Integrate All Factors
    integrated_alpha_factor = final_alpha_factor * weighted_combined
    
    # Add a smoothing factor to the Integrated Alpha Factor
    ema_smoothed = integrated_alpha_factor.ewm(alpha=0.3).mean()
    
    return ema_smoothed
