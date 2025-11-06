import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate basic volatility measures
    data['daily_range'] = data['high'] - data['low']
    data['open_close_range'] = data['close'] - data['open']
    data['range_utilization'] = data['open_close_range'] / data['daily_range']
    
    # Short-term volatility persistence (5-day)
    data['vol_momentum'] = (data['daily_range'] / data['daily_range'].shift(4)) - 1
    data['directional_vol_bias'] = (data['open_close_range'] / data['daily_range']) - (data['open_close_range'].shift(4) / data['daily_range'].shift(4))
    data['vol_volume_coupling'] = ((data['daily_range'] - data['daily_range'].shift(4)) * (data['volume'] - data['volume'].shift(4)))
    
    # Medium-term volatility acceleration (15-day)
    data['vol_expansion_rate'] = (data['daily_range'] / data['daily_range'].shift(14)) - 1
    data['asymmetric_vol_growth'] = ((data['high'] - data['open']) / (data['open'] - data['low']).replace(0, np.nan)) - ((data['high'].shift(14) - data['open'].shift(14)) / (data['open'].shift(14) - data['low'].shift(14)).replace(0, np.nan))
    data['regime_transition'] = np.sign(data['daily_range'] - data['daily_range'].shift(14)) * np.sign(data['volume'] - data['volume'].shift(14))
    
    # Intraday volatility structure
    data['opening_vol_premium'] = (data['high'] - data['open']) / (data['open'] - data['low']).replace(0, np.nan)
    data['closing_vol_compression'] = (data['close'] - data['low']) / (data['high'] - data['close']).replace(0, np.nan)
    data['daily_vol_asymmetry'] = (data['high'] - data['open'] - (data['open'] - data['low'])) / data['daily_range']
    
    # Volume-volatility dynamics
    data['vol_vol_ratio'] = (data['volume'] / data['daily_range']) - (data['volume'].shift(1) / data['daily_range'].shift(1))
    data['elasticity_momentum'] = ((data['volume'] / data['daily_range']) / (data['volume'].shift(1) / data['daily_range'].shift(1)).replace(0, np.nan)) - 1
    data['regime_specific_elasticity'] = np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(data['daily_range'] - data['daily_range'].shift(1))
    
    # Volume flow patterns
    data['concentrated_flow'] = ((data['volume'] > data['volume'].shift(1)) & (data['daily_range'] < data['daily_range'].shift(1))).astype(float)
    data['expanding_flow'] = ((data['volume'] > data['volume'].shift(1)) & (data['daily_range'] > data['daily_range'].shift(1))).astype(float)
    
    # Volume spike analysis
    data['vol_adj_volume_spike'] = ((data['volume'] > 1.8 * data['volume'].shift(1)) * (data['daily_range'] / data['daily_range'].shift(1))).fillna(0)
    data['directional_volume_spike'] = ((data['volume'] > 2.0 * data['volume'].shift(1)) * (data['open_close_range'] / data['daily_range'])).fillna(0)
    
    # Price efficiency measures
    data['directional_efficiency_momentum'] = data['range_utilization'] - data['range_utilization'].shift(1)
    data['efficiency_vol_correlation'] = np.sign(data['range_utilization']) * np.sign(data['daily_range'] - data['daily_range'].shift(1))
    
    # Opening gap efficiency
    data['gap_absorption'] = abs(data['open'] - data['close'].shift(1)) / data['daily_range']
    data['gap_directional_bias'] = (data['open'] - data['close'].shift(1)) / data['daily_range']
    
    # Closing momentum patterns
    data['closing_range_pos'] = (data['close'] - data['low']) / data['daily_range']
    data['closing_momentum_shift'] = data['closing_range_pos'] - data['closing_range_pos'].shift(1)
    data['end_of_day_efficiency'] = data['range_utilization'] * data['closing_vol_compression']
    
    # Multi-timeframe regime alignment
    data['short_medium_alignment'] = np.sign(data['vol_momentum']) * np.sign(data['vol_expansion_rate'])
    data['regime_confirmation'] = data['vol_volume_coupling'] * data['regime_transition']
    data['multi_scale_vol_convergence'] = data['vol_momentum'] * data['vol_expansion_rate']
    
    # Volume-volatility regime signals
    data['elasticity_regime_alignment'] = data['vol_vol_ratio'] * data['elasticity_momentum']
    
    # Adaptive momentum construction
    data['high_vol_momentum'] = data['range_utilization'] * (data['vol_momentum'] > 0.1)
    data['low_vol_momentum'] = data['range_utilization'] * (data['vol_momentum'] < -0.1)
    data['transition_regime_momentum'] = data['directional_efficiency_momentum'] * (abs(data['vol_momentum']) < 0.05)
    
    data['volume_enhanced_momentum'] = data['range_utilization'] * data['vol_vol_ratio']
    data['spike_confirmed_momentum'] = data['directional_volume_spike'] * data['closing_momentum_shift']
    
    # Composite alpha generation
    # Core volatility signals
    data['primary_vol_signal'] = data['vol_momentum'] * data['vol_volume_coupling']
    data['secondary_regime_signal'] = data['regime_transition'] * data['vol_vol_ratio']
    data['efficiency_core'] = data['range_utilization'] * data['directional_efficiency_momentum']
    
    # Calculate volume flow consistency (5-day window)
    vol_vol_sign = np.sign(data['regime_specific_elasticity'])
    data['volume_flow_consistency'] = vol_vol_sign.rolling(window=5, min_periods=1).apply(lambda x: len(x[x == x.iloc[-1]]) if len(x) > 0 else 0, raw=False)
    data['volume_core'] = data['elasticity_momentum'] * data['volume_flow_consistency']
    
    # Momentum enhancement layer
    data['regime_adaptive_momentum'] = data['high_vol_momentum'] * data['low_vol_momentum']
    data['volume_confirmed_enhancement'] = data['volume_enhanced_momentum'] * data['spike_confirmed_momentum']
    
    # Gap-adjusted momentum
    data['gap_adjusted_momentum'] = data['gap_directional_bias'] * data['gap_absorption']
    data['closing_enhanced_momentum'] = data['closing_range_pos'] * data['end_of_day_efficiency']
    data['multi_timeframe_momentum'] = data['short_medium_alignment'] * (data['range_utilization'] * data['vol_momentum'])
    
    data['efficiency_based_momentum'] = data['gap_adjusted_momentum'] * data['closing_enhanced_momentum']
    data['multi_scale_momentum'] = data['multi_timeframe_momentum'] * data['multi_scale_vol_convergence']
    
    # Final alpha construction
    data['core_vol_signals'] = data['primary_vol_signal'] * data['secondary_regime_signal'] * data['efficiency_core'] * data['volume_core']
    data['momentum_enhancement'] = data['regime_adaptive_momentum'] * data['volume_confirmed_enhancement'] * data['efficiency_based_momentum'] * data['multi_scale_momentum']
    
    data['base_alpha'] = data['core_vol_signals'] * data['momentum_enhancement']
    data['confirmation_signal'] = data['efficiency_core'] * data['volume_core']
    data['regime_adjustment'] = data['regime_adaptive_momentum'] * data['transition_regime_momentum']
    
    # Final composite alpha
    alpha = data['base_alpha'] * data['confirmation_signal'] * data['regime_adjustment']
    
    return alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
