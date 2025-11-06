import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Directional Volume Pressure Analysis
    # Intraday Pressure Imbalance
    upward_pressure = ((data['close'] - data['open']) * data['volume'] / 
                      (data['high'] - data['low'] + 1e-8) * np.sign(data['close'] - data['open']))
    downward_pressure = (np.abs(data['low'] - data['open']) * data['volume'] / 
                        (data['high'] - data['low'] + 1e-8) * np.sign(data['open'] - data['low']))
    net_pressure = upward_pressure - downward_pressure
    
    # Pressure Persistence Metrics
    pressure_direction = np.sign(net_pressure)
    consecutive_pressure = pressure_direction.rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and x.iloc[i] != 0]), 
        raw=False
    )
    
    pressure_3d_avg = net_pressure.rolling(window=3, min_periods=1).mean()
    pressure_10d_avg = net_pressure.rolling(window=10, min_periods=1).mean()
    pressure_magnitude_trend = pressure_3d_avg - pressure_10d_avg
    
    max_pressure_5d = net_pressure.rolling(window=5, min_periods=1).max()
    pressure_decay_rate = net_pressure / (max_pressure_5d + 1e-8)
    
    # Price Gap Momentum Dynamics
    # Gap Opening Analysis
    gap_strength = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    gap_filling_efficiency = ((data['close'] - data['open']) / 
                             (data['open'] - data['close'].shift(1) + 1e-8 * np.sign(data['open'] - data['close'].shift(1))))
    gap_filling_efficiency = gap_filling_efficiency.replace([np.inf, -np.inf], 0).fillna(0)
    gap_direction_persistence = np.sign(gap_strength) * np.sign(data['close'] - data['open'])
    
    # Gap-Volume Interaction
    volume_amplification = (data['volume'] / (data['volume'].shift(1) + 1e-8)) * np.abs(gap_strength)
    gap_support_resistance = ((data['high'] - data['open']) / 
                             (data['open'] - data['low'] + 1e-8) * np.sign(gap_strength))
    gap_support_resistance = gap_support_resistance.replace([np.inf, -np.inf], 0).fillna(0)
    gap_momentum_confirmation = (gap_strength * (data['close'] - data['open']) / 
                                (data['high'] - data['low'] + 1e-8))
    
    # Multi-Day Gap Patterns
    gap_sequence = gap_strength.rolling(window=3, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x.iloc[i]) == np.sign(x.iloc[i-1]) and x.iloc[i] != 0]), 
        raw=False
    )
    
    gap_reversal_count = gap_strength.rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x.iloc[i]) != np.sign(x.iloc[i-1])]), 
        raw=False
    )
    
    gap_momentum_accumulation = gap_strength.rolling(window=3, min_periods=1).sum()
    
    # Asymmetric Range Efficiency
    # Upper vs Lower Range Utilization
    upper_range_efficiency = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    lower_range_efficiency = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    range_asymmetry = upper_range_efficiency - lower_range_efficiency
    
    # Range Compression-Expansion Dynamics
    avg_range_5d = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    range_compression_ratio = (data['high'] - data['low']) / (avg_range_5d + 1e-8)
    expansion_direction_bias = ((data['close'] - data['open']) / 
                               (data['high'] - data['low'] + 1e-8) * range_compression_ratio)
    range_breakout_quality = (np.abs(data['close'] - data['open']) / 
                             (data['high'] - data['low'] + 1e-8) * range_compression_ratio)
    
    # Volume-Weighted Price Acceleration
    # Short-Term Acceleration Patterns
    momentum_3d = (data['close'] - data['close'].shift(3)) / (data['close'].shift(3) + 1e-8)
    volume_weighted_momentum = momentum_3d * (data['volume'] / (data['volume'].shift(1) + 1e-8))
    
    momentum_1d = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    acceleration_magnitude = ((momentum_1d - momentum_1d.shift(1)) * 
                             (data['volume'] / (data['volume'].shift(1) + 1e-8)))
    
    acceleration_consistency = (np.sign(acceleration_magnitude) * 
                               np.sign(acceleration_magnitude.shift(1)))
    
    # Multi-Scale Acceleration Alignment
    momentum_5d = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) + 1e-8)
    volume_trend = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    acceleration_divergence_3v5 = (momentum_3d - momentum_5d) * volume_trend
    
    acceleration_divergence = momentum_3d / (momentum_5d + 1e-8 * np.sign(momentum_5d))
    volume_confirmed_acceleration = acceleration_magnitude * volume_trend
    
    # Composite Alpha Integration
    # Normalize components
    pressure_component = (net_pressure.rolling(window=10, min_periods=1).mean() + 
                         pressure_magnitude_trend + pressure_decay_rate)
    pressure_component = pressure_component / (pressure_component.rolling(window=20, min_periods=1).std() + 1e-8)
    
    gap_component = (gap_momentum_confirmation + gap_support_resistance + 
                    gap_momentum_accumulation - gap_reversal_count)
    gap_component = gap_component / (gap_component.rolling(window=20, min_periods=1).std() + 1e-8)
    
    range_component = (range_asymmetry + expansion_direction_bias + range_breakout_quality)
    range_component = range_component / (range_component.rolling(window=20, min_periods=1).std() + 1e-8)
    
    acceleration_component = (volume_weighted_momentum + acceleration_divergence_3v5 + 
                            volume_confirmed_acceleration)
    acceleration_component = acceleration_component / (acceleration_component.rolling(window=20, min_periods=1).std() + 1e-8)
    
    # Dynamic Component Weighting based on pressure regime
    pressure_regime = np.where(np.abs(net_pressure) > net_pressure.rolling(window=20, min_periods=1).std(), 
                              'high', 
                              np.where(np.abs(net_pressure) < net_pressure.rolling(window=20, min_periods=1).std() * 0.5, 
                                      'low', 'normal'))
    
    composite_alpha = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if pressure_regime[i] == 'high':
            composite_alpha.iloc[i] = (0.45 * pressure_component.iloc[i] + 
                                      0.30 * gap_component.iloc[i] + 
                                      0.25 * range_component.iloc[i])
        elif pressure_regime[i] == 'low':
            composite_alpha.iloc[i] = (0.35 * range_component.iloc[i] + 
                                      0.40 * acceleration_component.iloc[i] + 
                                      0.25 * gap_component.iloc[i])
        else:  # normal
            composite_alpha.iloc[i] = (0.25 * pressure_component.iloc[i] + 
                                      0.25 * gap_component.iloc[i] + 
                                      0.25 * range_component.iloc[i] + 
                                      0.25 * acceleration_component.iloc[i])
    
    # Signal Quality Enhancement
    volume_confirmation = data['volume'] > data['volume'].rolling(window=10, min_periods=1).mean()
    range_compression_signal = range_compression_ratio < 0.8
    
    # Apply quality filters
    final_alpha = composite_alpha.copy()
    final_alpha[~volume_confirmation] = final_alpha[~volume_confirmation] * 0.5
    final_alpha[range_compression_signal] = final_alpha[range_compression_signal] * 1.2
    
    return final_alpha
