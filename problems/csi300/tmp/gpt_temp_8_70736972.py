import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Compute Amplitude-Weighted Momentum Acceleration
    # Bidirectional Amplitude Pressure
    upper_amplitude_pressure = (data['high'] - data['close']) * data['volume']
    lower_amplitude_pressure = (data['close'] - data['low']) * data['volume']
    
    # Amplitude Momentum Gap
    amplitude_momentum_gap = upper_amplitude_pressure - lower_amplitude_pressure
    
    # Momentum Acceleration
    short_term_momentum = data['close'] / data['close'].shift(2) - 1
    medium_term_momentum = data['close'] / data['close'].shift(7) - 1
    momentum_acceleration = short_term_momentum - medium_term_momentum
    
    # 2. Analyze Volume-Confirmed Breakout Patterns
    # Price Range Breakout Strength
    high_15d = data['high'].rolling(window=15, min_periods=1).max()
    low_15d = data['low'].rolling(window=15, min_periods=1).min()
    breakout_strength = (data['close'] - low_15d) / (high_15d - low_15d + 1e-8)
    
    # Breakout Momentum Persistence (3-day consistency)
    breakout_direction = np.sign(data['close'] - data['close'].shift(1))
    breakout_persistence = breakout_direction.rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0
    )
    
    # Volume Confirmation Intensity
    volume_intensity_ratio = data['volume'] / (data['high'] - data['low'] + 1e-8)
    median_volume_intensity = volume_intensity_ratio.rolling(window=10, min_periods=1).median()
    volume_acceleration = (data['volume'].rolling(window=5, min_periods=1).mean() / 
                          data['volume'].rolling(window=10, min_periods=1).mean())
    
    # Breakout-Volume Divergence (5-day correlation)
    breakout_volume_divergence = volume_intensity_ratio.rolling(window=5, min_periods=1).corr(
        breakout_strength.rolling(window=5, min_periods=1).mean()
    )
    
    # 3. Combine Amplitude Acceleration with Breakout Confirmation
    # Amplitude-Breakout Interaction
    amplitude_breakout_interaction = amplitude_momentum_gap * breakout_strength
    
    # Volume Confirmation Weighting
    volume_weighted_signal = (amplitude_breakout_interaction * 
                             volume_intensity_ratio / (median_volume_intensity + 1e-8) * 
                             volume_acceleration)
    
    # Momentum-Volume Convergence
    direction_alignment = (np.sign(amplitude_momentum_gap) * np.sign(breakout_strength))
    convergence_multiplier = np.where(direction_alignment > 0, 1.5, 0.8)
    convergence_signal = volume_weighted_signal * convergence_multiplier
    
    # 4. Filter with Amplitude-Persistence Validation
    # Amplitude Stability (3-day range consistency)
    amplitude_range = (data['high'] - data['low']).rolling(window=3, min_periods=1)
    amplitude_stability = 1 / (1 + amplitude_range.std() / (amplitude_range.mean() + 1e-8))
    
    # Momentum Acceleration Persistence (3-day direction consistency)
    acceleration_direction = np.sign(momentum_acceleration)
    acceleration_persistence = acceleration_direction.rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 0
    )
    
    # Apply Amplitude-Momentum Filter
    persistence_threshold = 0.6
    filtered_signal = (convergence_signal * amplitude_stability * 
                      np.where(acceleration_persistence >= persistence_threshold, 1, 0.3))
    
    # 5. Dynamic Signal Integration
    # Amplitude-Regime Adaptive Weighting
    amplitude_regime = (data['high'] - data['low']).rolling(window=10, min_periods=1).mean()
    amplitude_quantile = amplitude_regime.rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )
    
    # Regime-adaptive weights
    high_amplitude_weight = np.where(amplitude_quantile > 0.7, 1.2, 1.0)
    low_amplitude_weight = np.where(amplitude_quantile < 0.3, 1.3, 1.0)
    regime_weight = high_amplitude_weight * low_amplitude_weight
    
    # Volume-Intensity Final Adjustment
    volume_trend = data['volume'].rolling(window=3, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.mean() + 1e-8) if len(x) > 1 else 0
    )
    volume_breakout_multiplier = np.where(volume_trend > 0.1, 1.2, 1.0)
    
    # Generate Composite Predictive Signal
    composite_signal = (filtered_signal * regime_weight * 
                       volume_breakout_multiplier * momentum_acceleration)
    
    # 6. Generate Final Volume-Amplitude Acceleration Divergence
    # Apply Cubic Transformation
    final_factor = np.sign(composite_signal) * np.abs(composite_signal) ** (1/3)
    
    return final_factor
