import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Close Position in Range
    close_position_in_range = (df['close'] - df['low']) / high_low_range
    
    # Calculate Volume-Weighted Position
    volume_weighted_position = close_position_in_range * df['volume']
    
    # Apply Exponential Smoothing to Volume-Weighted Position
    smoothed_position = volume_weighted_position.ewm(alpha=0.2).mean()
    
    # Calculate Moving Average of Exponentially Smoothed Position
    smoothed_position_ma = smoothed_position.rolling(window=20).mean()
    
    # Calculate Short-Term Momentum (7-Day WMA)
    short_term_momentum = df['close'].rolling(window=7).apply(lambda x: np.average(x, weights=np.arange(1, 8)))
    
    # Calculate Long-Term Momentum (28-Day WMA)
    long_term_momentum = df['close'].rolling(window=28).apply(lambda x: np.average(x, weights=np.arange(1, 29)))
    
    # Calculate Momentum Difference
    raw_momentum_difference = short_term_momentum - long_term_momentum
    
    # Integrate Volume Information
    avg_volume_7_days = df['volume'].rolling(window=7).mean()
    inverse_avg_volume = 1 / (avg_volume_7_days + 1e-6)  # Add small constant to avoid division by zero
    integrated_momentum = raw_momentum_difference * inverse_avg_volume
    
    # Adjust for Volume Spikes
    volume_spike_factor = 0.6
    volume_ma_20_days = df['volume'].rolling(window=20).mean()
    volume_spike = (df['volume'] > 1.5 * volume_ma_20_days)
    adjusted_integrated_momentum = integrated_momentum.where(~volume_spike, volume_spike_factor * integrated_momentum)
    
    # Combine Factors
    final_alpha_factor = smoothed_position_ma + adjusted_integrated_momentum
    
    return final_alpha_factor
