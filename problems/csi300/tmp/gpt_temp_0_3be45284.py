import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Price-Volume Divergence
    # Short-term (5-day)
    price_change_5d = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    volume_change_5d = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    short_divergence = price_change_5d - volume_change_5d
    
    # Medium-term (10-day)
    price_change_10d = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    volume_change_10d = (data['volume'] - data['volume'].shift(10)) / data['volume'].shift(10)
    medium_divergence = price_change_10d - volume_change_10d
    
    # Divergence Difference
    divergence_diff = abs(short_divergence - medium_divergence) * np.sign(short_divergence) * np.sign(medium_divergence)
    
    # Fractal Market Analysis
    # Fractal Efficiency
    price_range_10d = abs(data['close'] - data['close'].shift(10))
    price_volatility_10d = data['close'].diff().abs().rolling(window=10, min_periods=1).sum()
    fractal_efficiency = price_range_10d / np.maximum(price_volatility_10d, 0.001)
    
    # Volume-Weighted Fractal
    volume_ratio = data['volume'] / np.maximum(data['volume'].shift(1), 0.001) - 1
    volume_weighted_fractal = fractal_efficiency * volume_ratio
    
    # Volume Acceleration Analysis
    # Volume Acceleration
    volume_acceleration = data['volume'] / np.maximum(data['volume'].shift(1), 0.001) - 1
    
    # Volume Acceleration Momentum
    volume_acceleration_lag = volume_acceleration.shift(1)
    volume_acceleration_momentum = (volume_acceleration - volume_acceleration_lag) / np.maximum(abs(volume_acceleration_lag), 0.001)
    
    # Volatility Regime Classification
    # Daily Volatility Range
    daily_vol_range = (data['high'] - data['low']) / np.maximum(data['close'], 0.001)
    
    # Volatility Persistence
    volatility_persistence = daily_vol_range.rolling(window=10, min_periods=1).std()
    
    # Directional Consistency (for High Efficiency regime)
    directional_consistency = data['close'].diff().rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) > 0 else 0, raw=False
    )
    
    # Regime Classification
    high_efficiency = (fractal_efficiency > 0.6) & (directional_consistency > 0)
    low_efficiency = (fractal_efficiency < 0.4) & (divergence_diff > 0)
    volatility_ratio = daily_vol_range / np.maximum(daily_vol_range.rolling(window=10, min_periods=1).mean(), 0.001)
    transition = (volume_acceleration_momentum > 2) & (volatility_ratio > 1.5)
    
    # Regime-Adaptive Signal Generation
    # Volatility-Adjusted Divergence
    volatility_adjusted_divergence = divergence_diff / np.maximum(volatility_persistence, 0.001)
    
    # Fractal-Enhanced Signal
    fractal_enhanced_signal = volatility_adjusted_divergence * volume_weighted_fractal
    
    # Regime-Based Weighting
    regime_weight = np.ones_like(fractal_enhanced_signal)
    regime_weight[high_efficiency] = 1.5
    regime_weight[low_efficiency] = 0.7
    regime_weight[transition] = volume_acceleration_momentum[transition]
    
    # Volume-Confirmed Output
    volume_confirmed_signal = regime_weight * fractal_enhanced_signal * np.sign(volume_acceleration)
    
    # Return the final factor series
    return volume_confirmed_signal
