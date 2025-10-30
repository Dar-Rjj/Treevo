import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Calculate Volatility-Adjusted Price Momentum
    # Compute Short-Term Price Momentum
    price_return_5d = df['close'] / df['close'].shift(5) - 1
    price_return_10d = df['close'] / df['close'].shift(10) - 1
    
    # Normalize by Price Volatility
    high_low_range = df['high'] - df['low']
    volatility_20d = high_low_range.rolling(window=20).mean()
    
    volatility_adjusted_momentum_5d = price_return_5d / volatility_20d
    volatility_adjusted_momentum_10d = price_return_10d / volatility_20d
    
    # Combine both momentum periods
    volatility_adjusted_momentum = (volatility_adjusted_momentum_5d + volatility_adjusted_momentum_10d) / 2
    
    # Calculate Volume Trend Strength
    # Compute Volume Momentum
    volume_change_5d = df['volume'] / df['volume'].shift(5) - 1
    volume_change_10d = df['volume'] / df['volume'].shift(10) - 1
    volume_acceleration = volume_change_5d / volume_change_10d
    
    # Assess Volume Trend Consistency
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    volume_above_avg_count = (df['volume'] > volume_20d_avg.shift(1)).rolling(window=10).sum()
    
    # Calculate volume trend slope using linear regression
    def volume_slope(volume_series):
        if len(volume_series) < 2:
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    volume_trend_slope = df['volume'].rolling(window=10).apply(volume_slope, raw=False)
    
    # Combine volume trend components
    volume_trend_strength = (
        volume_acceleration.fillna(0) + 
        (volume_above_avg_count / 10) + 
        (volume_trend_slope / df['volume'].rolling(window=10).mean())
    ) / 3
    
    # Generate Divergence Signal
    # Compare Directional Alignment
    price_volume_aligned = np.sign(volatility_adjusted_momentum) == np.sign(volume_trend_strength)
    directional_multiplier = np.where(price_volume_aligned, 1, -1)
    
    # Assess magnitude consistency
    price_magnitude = np.abs(volatility_adjusted_momentum)
    volume_magnitude = np.abs(volume_trend_strength)
    magnitude_ratio = price_magnitude / (volume_magnitude + 1e-8)
    magnitude_consistency = np.where(
        (magnitude_ratio > 0.5) & (magnitude_ratio < 2.0), 
        1.0, 
        0.5
    )
    
    # Apply Volume Confirmation Filter
    divergence_signal = (
        volatility_adjusted_momentum * 
        volume_trend_strength * 
        directional_multiplier * 
        magnitude_consistency
    )
    
    return divergence_signal
