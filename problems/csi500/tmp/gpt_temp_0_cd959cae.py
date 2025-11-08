import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Parameters
    n_days = 20
    regime_lookback = 50
    acceleration_period = 5
    
    # Fractal Efficiency Calculation
    def calculate_fractal_efficiency(high, low, close, n):
        path_length = 0
        for i in range(1, n):
            path_length += abs(high.iloc[-i] - low.iloc[-i])
        
        straight_distance = abs(close.iloc[-1] - close.iloc[-n])
        efficiency = straight_distance / path_length if path_length > 0 else 0
        return efficiency
    
    # Volume Fractal Dimension Analysis
    def calculate_volume_fractal(volume, n):
        volume_changes = volume.diff().abs()
        volume_path = volume_changes.rolling(window=n).sum()
        volume_range = volume.rolling(window=n).max() - volume.rolling(window=n).min()
        fractal_dim = np.log(volume_path) / np.log(volume_range) if volume_range.iloc[-1] > 0 else 1
        return fractal_dim
    
    # Calculate rolling fractal efficiency
    fractal_efficiency = pd.Series(index=data.index, dtype=float)
    for i in range(n_days, len(data)):
        fractal_efficiency.iloc[i] = calculate_fractal_efficiency(
            data['high'].iloc[:i+1], 
            data['low'].iloc[:i+1], 
            data['close'].iloc[:i+1], 
            n_days
        )
    
    # Calculate volume fractal dimension
    volume_fractal = calculate_volume_fractal(data['volume'], n_days)
    
    # Regime-Switching Acceleration Model
    # Efficiency regime identification
    efficiency_ma = fractal_efficiency.rolling(window=regime_lookback).mean()
    efficiency_std = fractal_efficiency.rolling(window=regime_lookback).std()
    efficiency_zscore = (fractal_efficiency - efficiency_ma) / efficiency_std
    
    high_efficiency_regime = (efficiency_zscore > 1).astype(int)
    low_efficiency_regime = (efficiency_zscore < -1).astype(int)
    normal_regime = ((efficiency_zscore >= -1) & (efficiency_zscore <= 1)).astype(int)
    
    # Price acceleration (second derivative)
    price_first_deriv = data['close'].diff()
    price_acceleration = price_first_deriv.diff().rolling(window=acceleration_period).mean()
    
    # Volume acceleration (second derivative)
    volume_first_deriv = data['volume'].diff()
    volume_acceleration = volume_first_deriv.diff().rolling(window=acceleration_period).mean()
    
    # Normalize accelerations
    price_acceleration_norm = (price_acceleration - price_acceleration.rolling(window=regime_lookback).mean()) / price_acceleration.rolling(window=regime_lookback).std()
    volume_acceleration_norm = (volume_acceleration - volume_acceleration.rolling(window=regime_lookback).mean()) / volume_acceleration.rolling(window=regime_lookback).std()
    
    # Adaptive Signal Generation
    # High efficiency regime: momentum acceleration
    high_eff_signal = price_acceleration_norm * (1 + volume_acceleration_norm)
    
    # Low efficiency regime: mean reversion + volume complexity
    price_zscore = (data['close'] - data['close'].rolling(window=n_days).mean()) / data['close'].rolling(window=n_days).std()
    low_eff_signal = -price_zscore * volume_fractal
    
    # Normal regime: balanced approach
    normal_signal = (price_acceleration_norm * 0.5) + (volume_fractal * 0.3) - (price_zscore * 0.2)
    
    # Combine signals based on regime
    final_signal = (
        high_efficiency_regime * high_eff_signal +
        low_efficiency_regime * low_eff_signal +
        normal_regime * normal_signal
    )
    
    # Apply efficiency weighting
    efficiency_weight = fractal_efficiency.rolling(window=5).mean()
    weighted_signal = final_signal * efficiency_weight
    
    return weighted_signal
