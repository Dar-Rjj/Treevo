import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Price-Velocity Calculation
    # 3-day rate of change (short-term velocity)
    velocity_3d = data['close'].pct_change(periods=3)
    
    # 5-day rate of change (medium-term velocity)
    velocity_5d = data['close'].pct_change(periods=5)
    
    # 10-day rate of change (long-term velocity)
    velocity_10d = data['close'].pct_change(periods=10)
    
    # Apply exponential decay weighting
    weights = np.array([0.5, 0.3, 0.2])  # Recent velocities weighted higher
    weighted_velocity = (velocity_3d * weights[0] + 
                        velocity_5d * weights[1] + 
                        velocity_10d * weights[2])
    
    # Velocity Divergence Calculation
    # Price and volume 3-day rate of change
    price_roc_3d = data['close'].pct_change(periods=3)
    volume_roc_3d = data['volume'].pct_change(periods=3)
    
    # Rolling 5-day correlation between price and volume velocities
    correlation = price_roc_3d.rolling(window=5).corr(volume_roc_3d)
    
    # Expected correlation (historical mean)
    expected_corr = correlation.rolling(window=20, min_periods=5).mean()
    
    # Divergence magnitude with sign adjustment
    divergence = (correlation - expected_corr).abs()
    divergence_sign = np.sign(price_roc_3d)
    signed_divergence = divergence * divergence_sign
    
    # Smoothing filter
    smoothed_divergence = signed_divergence.rolling(window=3).mean()
    
    # Volume Confirmation Strength
    # Volume acceleration
    volume_accel_3d = data['volume'].pct_change(periods=3)
    volume_accel_5d = data['volume'].pct_change(periods=5)
    
    # Volume trend consistency
    volume_10d_avg = data['volume'].rolling(window=10).mean()
    volume_ratio = data['volume'] / volume_10d_avg
    
    # Persistence of volume acceleration (positive acceleration days in last 5)
    volume_accel_pos = (volume_accel_3d > 0).astype(int)
    volume_persistence = volume_accel_pos.rolling(window=5).sum() / 5
    
    # Volume confirmation strength composite
    volume_strength = (volume_accel_3d.abs() * 0.4 + 
                      volume_accel_5d.abs() * 0.3 + 
                      volume_ratio * 0.2 + 
                      volume_persistence * 0.1)
    
    # Volatility Regime Identification
    # Daily range
    daily_range = data['high'] - data['low']
    
    # Rolling volatility (15-day standard deviation of daily ranges)
    volatility = daily_range.rolling(window=15).std()
    
    # Historical volatility percentiles
    vol_percentile = volatility.rolling(window=60, min_periods=20).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 60)) * 2 + 
                 (x.iloc[-1] > np.percentile(x.dropna(), 40)) * 1, 
        raw=False
    )
    
    # Regime classification
    high_vol_regime = (vol_percentile == 3).astype(int)  # Above 60th percentile
    medium_vol_regime = (vol_percentile == 1).astype(int)  # 40th-60th percentile
    low_vol_regime = (vol_percentile == 0).astype(int)  # Below 40th percentile
    
    # Price-level adjustment (support/resistance proximity)
    # Recent high and low (10-day window)
    recent_high = data['high'].rolling(window=10).max()
    recent_low = data['low'].rolling(window=10).min()
    
    # Distance to support and resistance
    dist_to_resistance = (recent_high - data['close']) / data['close']
    dist_to_support = (data['close'] - recent_low) / data['close']
    
    # Price level adjustment factor (enhance signal near boundaries)
    price_adjustment = 1 + (np.minimum(dist_to_resistance, dist_to_support) * 2)
    
    # Adaptive Composite Signal Generation
    # Combine velocity divergence with volume confirmation
    base_signal = smoothed_divergence * volume_strength
    
    # Regime-specific adjustments
    high_vol_signal = base_signal.rolling(window=3).mean() * (1 / (volatility + 0.001))
    medium_vol_signal = base_signal.rolling(window=5).mean()
    low_vol_signal = base_signal.rolling(window=10).mean()
    
    # Apply regime-specific signals
    regime_adjusted_signal = (
        high_vol_signal * high_vol_regime +
        medium_vol_signal * medium_vol_regime +
        low_vol_signal * low_vol_regime
    )
    
    # Final signal with price-level adjustment
    final_signal = regime_adjusted_signal * price_adjustment
    
    return final_signal
