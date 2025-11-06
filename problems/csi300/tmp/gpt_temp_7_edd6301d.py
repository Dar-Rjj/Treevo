import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Basic price and volume calculations
    data['daily_range_utilization'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['volume_per_unit_range'] = data['volume'] / (data['high'] - data['low'] + 0.001)
    data['close_position_in_range'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 0.001)
    data['open_position_in_range'] = (data['open'] - data['low']) / (data['high'] - data['low'] + 0.001)
    
    # Range efficiency persistence
    data['high_efficiency_flag'] = (data['daily_range_utilization'] > 0.7).astype(int)
    data['range_efficiency_persistence'] = data['high_efficiency_flag'].groupby(data.index).expanding().apply(
        lambda x: (x == 1).cumsum().iloc[-1] if (x == 1).any() else 0, raw=False
    ).reset_index(level=0, drop=True)
    
    # Range efficiency momentum
    data['range_efficiency_momentum'] = data['daily_range_utilization'] - data['daily_range_utilization'].shift(1)
    
    # Multi-day range calculations
    data['range_3day_expansion'] = (data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3) + 0.001)
    
    # Range efficiency consistency
    data['range_efficiency_consistency'] = data['daily_range_utilization'].rolling(window=5, min_periods=3).std()
    
    # Volume-range efficiency
    data['volume_range_efficiency'] = data['daily_range_utilization'] * data['volume_per_unit_range']
    
    # Amplitude-volume momentum divergence
    data['range_change'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001) - 1
    data['volume_change'] = data['volume'] / data['volume'].shift(1) - 1
    data['amplitude_volume_divergence'] = data['range_change'] - data['volume_change']
    
    # Cross-timeframe range efficiency
    data['range_util_3day_avg'] = data['daily_range_utilization'].rolling(window=3, min_periods=2).mean()
    data['range_util_10day_avg'] = data['daily_range_utilization'].rolling(window=10, min_periods=5).mean()
    data['range_efficiency_gap'] = data['range_util_3day_avg'] - data['range_util_10day_avg']
    
    # Market microstructure calculations
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['closing_pressure'] = (data['close'] - data['open']) / data['open']
    data['pressure_consistency'] = np.sign(data['opening_pressure']) * np.sign(data['closing_pressure'])
    
    # Position shift
    data['position_shift'] = data['close_position_in_range'] - data['open_position_in_range']
    
    # Range imbalance
    data['upper_range_dominance'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 0.001)
    data['lower_range_dominance'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 0.001)
    data['range_imbalance_score'] = data['upper_range_dominance'] - data['lower_range_dominance']
    
    # Microstructure efficiency
    data['position_volume_alignment'] = data['position_shift'] * data['volume_per_unit_range']
    data['imbalance_pressure_correlation'] = data['range_imbalance_score'] * data['pressure_consistency']
    
    # Microstructure acceleration (simplified)
    data['microstructure_acceleration'] = data['pressure_consistency'] * data['position_shift']
    
    # Regime detection
    data['high_amplitude_efficiency'] = ((data['daily_range_utilization'] > 0.8) & 
                                       (data['range_efficiency_persistence'] >= 2)).astype(int)
    data['efficiency_strength'] = data['daily_range_utilization'] * data['range_efficiency_persistence']
    
    data['low_amplitude_efficiency'] = ((data['daily_range_utilization'] < 0.2) & 
                                      (data['range_efficiency_momentum'].abs() < 0.1)).astype(int)
    
    # Inefficiency persistence
    data['low_efficiency_flag'] = (data['daily_range_utilization'] < 0.3).astype(int)
    data['inefficiency_persistence'] = data['low_efficiency_flag'].groupby(data.index).expanding().apply(
        lambda x: (x == 1).cumsum().iloc[-1] if (x == 1).any() else 0, raw=False
    ).reset_index(level=0, drop=True)
    
    # High microstructure activity
    data['high_microstructure_activity'] = ((data['microstructure_acceleration'].abs() > 0.1) & 
                                          (data['volume_per_unit_range'] > data['volume_per_unit_range'].rolling(20).mean())).astype(int)
    data['activity_strength'] = data['microstructure_acceleration'].abs() * data['volume_per_unit_range']
    
    # Low microstructure activity
    data['low_microstructure_activity'] = ((data['microstructure_acceleration'].abs() < 0.05) & 
                                         (data['volume_per_unit_range'] < data['volume_per_unit_range'].rolling(20).mean() * 0.5)).astype(int)
    
    # Regime classification
    data['high_efficiency_high_activity'] = (data['high_amplitude_efficiency'] & data['high_microstructure_activity']).astype(int)
    data['low_efficiency_low_activity'] = (data['low_amplitude_efficiency'] & data['low_microstructure_activity']).astype(int)
    data['mixed_regime'] = (~data['high_efficiency_high_activity'] & ~data['low_efficiency_low_activity']).astype(int)
    
    # Strategy components
    # High efficiency strategy
    data['range_util_momentum_signal'] = data['range_efficiency_momentum'] * data['efficiency_strength']
    data['volume_amplitude_coordination'] = data['volume_range_efficiency'] * data['daily_range_utilization']
    data['efficiency_breakout_detection'] = data['range_efficiency_gap'] * data['range_3day_expansion']
    data['high_efficiency_strategy'] = (data['range_util_momentum_signal'] + 
                                      data['volume_amplitude_coordination'] + 
                                      data['efficiency_breakout_detection']) / 3
    
    # Low efficiency strategy
    data['range_compression_breakout'] = data['range_3day_expansion'] * data['inefficiency_persistence']
    data['volume_amplitude_divergence_signal'] = data['amplitude_volume_divergence'] * data['daily_range_utilization']
    data['microstructure_pressure_buildup'] = data['pressure_consistency'] * data['inefficiency_persistence']
    data['low_efficiency_strategy'] = (data['range_compression_breakout'] + 
                                     data['volume_amplitude_divergence_signal'] + 
                                     data['microstructure_pressure_buildup']) / 3
    
    # Transition regime strategy
    data['high_efficiency_weight'] = data['daily_range_utilization'] - 0.5
    data['low_efficiency_weight'] = 0.5 - data['daily_range_utilization']
    data['blended_amplitude_signal'] = (data['high_efficiency_weight'].clip(lower=0) * data['high_efficiency_strategy'] + 
                                      data['low_efficiency_weight'].clip(lower=0) * data['low_efficiency_strategy'])
    
    # Regime-adaptive factor
    data['regime_momentum'] = data['daily_range_utilization'] * data['microstructure_acceleration']
    
    # Final factor construction
    data['base_factor'] = np.where(
        data['high_efficiency_high_activity'] == 1,
        data['high_efficiency_strategy'],
        np.where(
            data['low_efficiency_low_activity'] == 1,
            data['low_efficiency_strategy'],
            data['blended_amplitude_signal']
        )
    )
    
    # Apply transition enhancement and microstructure confirmation
    data['enhanced_factor'] = data['base_factor'] * (1 + data['regime_momentum'])
    data['final_factor'] = data['enhanced_factor'] * (1 + data['pressure_consistency'] * 0.1)
    
    # Clean up and return
    result = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
