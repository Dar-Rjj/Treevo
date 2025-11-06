import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price-Volume Divergence Detection
    data['price_direction'] = np.sign(data['close'] - data['close'].shift(1))
    data['volume_direction'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['directional_alignment'] = data['price_direction'] * data['volume_direction']
    
    data['price_magnitude'] = np.abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['volume_magnitude'] = data['volume'] / data['volume'].shift(1) - 1
    data['magnitude_divergence'] = data['price_magnitude'] - data['volume_magnitude']
    
    data['divergence_strength'] = np.abs(data['directional_alignment']) * np.abs(data['magnitude_divergence'])
    data['divergence_persistence'] = data['directional_alignment'].rolling(window=3, min_periods=1).apply(lambda x: (x < 0).sum())
    data['divergence_acceleration'] = data['magnitude_divergence'] - data['magnitude_divergence'].shift(1)
    
    # Regime Transition Identification
    data['volatility_level'] = (data['high'] - data['low']) / data['close']
    data['volatility_change'] = data['volatility_level'] - data['volatility_level'].shift(1)
    data['regime_shift'] = np.sign(data['volatility_change']) * np.abs(data['volatility_change'])
    
    data['volume_level'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_regime_change'] = data['volume_level'] - data['volume_level'].shift(1)
    data['volume_regime_shift'] = np.sign(data['volume_regime_change']) * np.abs(data['volume_regime_change'])
    
    data['vol_vol_regime_alignment'] = data['regime_shift'] * data['volume_regime_shift']
    data['regime_transition_strength'] = np.abs(data['regime_shift']) + np.abs(data['volume_regime_shift'])
    data['transition_quality'] = data['vol_vol_regime_alignment'] * data['regime_transition_strength']
    
    # Gap Analysis & Fade Detection
    data['gap_size'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_direction'] = np.sign(data['gap_size'])
    data['gap_persistence'] = data['close'] - data['open']
    
    data['fade_magnitude'] = data['gap_persistence'] / data['gap_size'].replace(0, np.nan)
    data['fade_direction'] = np.sign(data['fade_magnitude']) * data['gap_direction']
    data['fade_quality'] = np.abs(data['fade_magnitude']) * (data['volume'] / data['volume'].shift(1))
    
    # Intraday Pressure Dynamics
    data['morning_pressure'] = (data['high'] - data['open']) * data['volume']
    data['afternoon_pressure'] = (data['close'] - data['low']) * data['volume']
    data['session_pressure_diff'] = data['morning_pressure'] - data['afternoon_pressure']
    
    data['pressure_reversal'] = np.sign(data['session_pressure_diff'] - data['session_pressure_diff'].shift(1))
    data['reversal_strength'] = np.abs(data['session_pressure_diff'] - data['session_pressure_diff'].shift(1))
    data['reversal_quality'] = data['reversal_strength'] * (data['volume'] / data['volume'].shift(1))
    
    # Range Expansion-Contraction Cycles
    data['daily_range'] = data['high'] - data['low']
    data['range_change'] = data['daily_range'] - data['daily_range'].shift(1)
    data['range_momentum'] = data['range_change'] - data['range_change'].shift(1)
    
    data['range_efficiency'] = data['daily_range'] / np.abs(data['close'] - data['open']).replace(0, np.nan)
    data['expansion_phase'] = np.sign(data['range_change']) * data['range_momentum']
    data['cycle_quality'] = data['expansion_phase'] * data['range_efficiency']
    
    # Multi-Timeframe Divergence Integration
    data['st_price_change'] = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    data['st_volume_change'] = data['volume'] / data['volume'].rolling(window=3, min_periods=1).mean() - 1
    data['st_divergence'] = data['st_price_change'] - data['st_volume_change']
    
    data['mt_price_change'] = (data['close'] - data['close'].shift(7)) / data['close'].shift(7)
    data['mt_volume_change'] = data['volume'] / data['volume'].rolling(window=8, min_periods=1).mean() - 1
    data['mt_divergence'] = data['mt_price_change'] - data['mt_volume_change']
    
    data['divergence_alignment'] = np.sign(data['st_divergence']) * np.sign(data['mt_divergence'])
    data['divergence_momentum'] = data['st_divergence'] - data['mt_divergence']
    data['multi_timeframe_quality'] = data['divergence_alignment'] * np.abs(data['divergence_momentum'])
    
    # Composite Factor Construction
    data['primary_divergence'] = data['directional_alignment'] * data['magnitude_divergence']
    data['multi_timeframe_divergence'] = data['primary_divergence'] * data['multi_timeframe_quality']
    data['regime_enhanced_divergence'] = data['multi_timeframe_divergence'] * data['transition_quality']
    
    data['gap_divergence'] = data['fade_quality'] * data['directional_alignment']
    data['pressure_divergence'] = data['reversal_quality'] * data['magnitude_divergence']
    data['combined_dynamics'] = data['gap_divergence'] * data['pressure_divergence']
    
    data['range_divergence'] = data['cycle_quality'] * data['divergence_strength']
    data['cycle_divergence_alignment'] = data['range_divergence'] * data['divergence_persistence']
    data['validated_signal'] = data['cycle_divergence_alignment'] * data['combined_dynamics']
    
    # Adaptive Signal Synthesis
    data['high_volatility_signal'] = data['validated_signal'] * data['regime_shift']
    data['low_volatility_signal'] = data['validated_signal'] * -np.abs(data['regime_shift'])
    data['volume_expansion_signal'] = data['validated_signal'] * data['volume_regime_shift']
    data['volume_contraction_signal'] = data['validated_signal'] * -np.abs(data['volume_regime_shift'])
    
    data['divergence_expansion'] = data['high_volatility_signal'] * data['expansion_phase']
    data['divergence_contraction'] = data['low_volatility_signal'] * -np.sign(data['range_change']) * data['range_momentum']
    data['gap_fade_regime'] = data['volume_expansion_signal'] * data['fade_direction']
    data['pressure_reversal_regime'] = data['volume_contraction_signal'] * data['pressure_reversal']
    
    # Risk & Quality Filters
    data['divergence_quality_floor'] = np.maximum(np.abs(data['divergence_strength']), 0.01)
    data['regime_consistency'] = np.sign(data['regime_shift']) * np.sign(data['volume_regime_shift'])
    data['volume_stability'] = 1 - np.abs(data['volume_level'] - 1)
    data['signal_quality'] = data['regime_consistency'] * data['volume_stability']
    
    # Final Alpha Output
    alpha = (data['divergence_expansion'] + data['divergence_contraction'] + 
             data['gap_fade_regime'] + data['pressure_reversal_regime']) * data['signal_quality']
    
    return alpha.fillna(0)
