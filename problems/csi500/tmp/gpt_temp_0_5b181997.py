import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Component
    # Short-term momentum (5-day return)
    short_momentum = data['close'] / data['close'].shift(5) - 1
    
    # Medium-term momentum (20-day return)
    medium_momentum = data['close'] / data['close'].shift(20) - 1
    
    # Volume Confirmation
    # Calculate 5-day volume slope using linear regression
    def volume_slope(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(5)
        y = volume_series.values
        return np.polyfit(x, y, 1)[0]
    
    # Apply rolling volume slope calculation
    volume_trend = data['volume'].rolling(window=5).apply(volume_slope, raw=False)
    
    # Volume-Price Alignment
    # Check if momentum signs align with volume trend
    momentum_alignment = ((short_momentum > 0) & (volume_trend > 0)) | ((short_momentum < 0) & (volume_trend < 0))
    
    # Divergence Signal
    # Identify momentum divergence (short-term vs medium-term moving in opposite directions)
    momentum_divergence = (short_momentum * medium_momentum) < 0
    
    # Strength of divergence (absolute difference between momentum values)
    divergence_strength = abs(short_momentum - medium_momentum)
    
    # Volume Validation - check if volume trend confirms the short-term momentum
    volume_confirmation = momentum_alignment & (abs(volume_trend) > volume_trend.rolling(20).mean())
    
    # Generate final alpha signal
    # Positive when: divergence exists, volume confirms short-term momentum, and divergence is strong
    alpha_signal = momentum_divergence & volume_confirmation & (divergence_strength > divergence_strength.rolling(20).mean())
    
    # Convert boolean to numeric and scale by divergence strength
    final_factor = alpha_signal.astype(float) * divergence_strength * np.sign(short_momentum)
    
    # Fill NaN values with 0
    final_factor = final_factor.fillna(0)
    
    return final_factor
