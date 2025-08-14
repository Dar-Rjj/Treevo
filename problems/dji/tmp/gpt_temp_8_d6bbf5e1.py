import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-to-Low Price Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Price Range Momentum
    price_range_momentum = high_low_range - high_low_range.shift(1)
    
    # Adjust by Volume and Price Change
    volume_change = df['volume'] - df['volume'].shift(1)
    price_change = df['close'] - df['open'].shift(1)
    
    # Integrate Momentum, Volume, and Price Change
    integrated_factor = (price_range_momentum * volume_change) + price_change
    
    # Confirm with Volume
    volume_change_confirmed = volume_change + 1  # Ensure positive adjusted volume change
    combined_factor = integrated_factor / volume_change_confirmed
    
    # Smoothing Filter
    ema_combined_factor = combined_factor.ewm(span=5, adjust=False).mean()
    
    # Calculate Adjusted Daily Return
    daily_return = df['close'] - df['close'].shift(1)
    price_gap = df['open'] - df['close'].shift(1)
    adjusted_daily_return = daily_return - np.where(price_gap > 0, price_gap, -price_gap)
    
    # Integrate Smoothed Volume-Adjusted High-Low Spread with Adjusted Daily Return
    integrated_smoothed_factor = adjusted_daily_return * ema_combined_factor
    
    # Adjust by Volume Change
    volume_change_final = df['volume'] / df['volume'].shift(1)
    
    # Final Integration
    final_factor = integrated_smoothed_factor * volume_change_final
    
    # Ensure Positive Factor
    final_factor = np.where(final_factor < 0, 0, final_factor)
    
    return final_factor
