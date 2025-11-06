import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility State Identification
    data['intraday_range'] = data['high'] - data['low']
    data['prev_intraday_range'] = data['intraday_range'].shift(1)
    data['prev_high_low'] = data['high'].shift(1) - data['low'].shift(1)
    
    # Volatility Expansion Detection
    data['intraday_vol_surge'] = data['intraday_range'] / data['prev_intraday_range']
    data['gap_volatility'] = np.abs(data['open'] - data['close'].shift(1)) / data['prev_high_low']
    
    # Volatility Breakout Score
    data['vol_breakout_condition'] = (data['intraday_range'] > 1.5 * data['prev_intraday_range']).astype(int)
    data['volatility_breakout_score'] = data['vol_breakout_condition'].rolling(window=5, min_periods=1).sum()
    
    # Volatility Contraction Patterns
    data['range_4'] = (data['high'] - data['low']).rolling(window=5).apply(lambda x: x.iloc[0] if len(x) == 5 else np.nan)
    data['range_compression_ratio'] = data['intraday_range'] / data['range_4']
    
    # Volatility Drying Signal
    contraction_condition = data['intraday_range'] < 0.8 * data['prev_intraday_range']
    data['volatility_drying_signal'] = np.where(contraction_condition, data['volume'] / data['volume'].shift(1), 0)
    
    # Contraction Persistence
    data['contraction_condition'] = (data['intraday_range'] < data['prev_intraday_range']).astype(int)
    data['contraction_persistence'] = data['contraction_condition'].rolling(window=5, min_periods=1).sum()
    
    # Volatility Regime Classification
    data['expansion_regime_score'] = data['volatility_breakout_score'] / 4
    data['contraction_regime_score'] = data['contraction_persistence'] / 4
    data['regime_transition_signal'] = data['expansion_regime_score'] - data['contraction_regime_score']
    
    # Price-Volume Divergence Analysis
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['volume_change'] = data['volume'] - data['volume'].shift(1)
    
    # Volume-Price Direction Mismatch
    mismatch_condition = np.sign(data['price_change']) != np.sign(data['volume_change'])
    data['negative_divergence'] = np.where(mismatch_condition, data['price_change'] / data['volume'], 0)
    
    match_condition = np.sign(data['price_change']) == np.sign(data['volume_change'])
    data['positive_divergence'] = np.where(match_condition, data['price_change'] / data['volume'], 0)
    
    data['divergence_intensity'] = np.abs(data['negative_divergence'] - data['positive_divergence'])
    
    # Volume-Volatility Dislocation
    high_vol_low_vol_condition = (data['volume'] > data['volume'].shift(1)) & (data['intraday_range'] < data['prev_intraday_range'])
    data['high_volume_low_volatility'] = np.where(high_vol_low_vol_condition, data['volume'] / data['intraday_range'], 0)
    
    low_vol_high_vol_condition = (data['volume'] < data['volume'].shift(1)) & (data['intraday_range'] > data['prev_intraday_range'])
    data['low_volume_high_volatility'] = np.where(low_vol_high_vol_condition, data['intraday_range'] / data['volume'], 0)
    
    data['dislocation_score'] = data['high_volume_low_volatility'] - data['low_volume_high_volatility']
    
    # Divergence Pattern Recognition
    data['divergence_condition'] = (data['divergence_intensity'] > 0).astype(int)
    data['divergence_persistence'] = data['divergence_condition'].rolling(window=5, min_periods=1).sum()
    
    data['dislocation_trend'] = data['dislocation_score'].rolling(window=5, min_periods=1).mean()
    data['divergence_quality'] = data['divergence_persistence'] * data['dislocation_trend']
    
    # Multi-Timeframe Regime Alignment
    data['immediate_regime_strength'] = data['regime_transition_signal'] * data['price_change']
    data['volatility_momentum'] = (data['intraday_range'] / data['prev_intraday_range']) * data['regime_transition_signal']
    data['short_term_alignment'] = data['immediate_regime_strength'] * data['volatility_momentum']
    
    data['weekly_regime_trend'] = data['regime_transition_signal'].rolling(window=5, min_periods=1).mean()
    
    # Volume Regime Correlation
    volume_window = data['volume'].rolling(window=5, min_periods=1)
    regime_window = data['regime_transition_signal'].rolling(window=5, min_periods=1)
    data['volume_regime_correlation'] = (volume_window.apply(lambda x: (x * data['regime_transition_signal'].loc[x.index]).sum()) / 
                                       volume_window.sum()).fillna(0)
    
    data['medium_term_convergence'] = data['weekly_regime_trend'] * data['volume_regime_correlation']
    
    # Multi-scale Integration
    data['timeframe_synchronization'] = data['short_term_alignment'] * data['medium_term_convergence']
    data['regime_quality_metric'] = 1 / (1 + np.abs(data['regime_transition_signal']))
    data['integrated_regime_signal'] = data['timeframe_synchronization'] * data['regime_quality_metric']
    
    # Adaptive Divergence Enhancement
    data['expansion_divergence_multiplier'] = 1 + data['expansion_regime_score']
    data['contraction_divergence_multiplier'] = 1 + data['contraction_regime_score']
    data['transition_divergence_multiplier'] = 1 + np.abs(data['regime_transition_signal'])
    
    data['expansion_divergence_signal'] = data['divergence_intensity'] * data['expansion_divergence_multiplier']
    data['contraction_divergence_signal'] = data['dislocation_score'] * data['contraction_divergence_multiplier']
    data['transition_divergence_boost'] = data['divergence_quality'] * data['transition_divergence_multiplier']
    
    # Dynamic Regime Routing
    def volatility_regime_detection(row):
        if row['volatility_breakout_score'] > 2:
            return 'Expansion'
        elif row['contraction_persistence'] > 2:
            return 'Contraction'
        else:
            return 'Transition'
    
    data['volatility_regime'] = data.apply(volatility_regime_detection, axis=1)
    
    def adaptive_signal_selection(row):
        if row['volatility_regime'] == 'Expansion':
            return row['expansion_divergence_signal']
        elif row['volatility_regime'] == 'Contraction':
            return row['contraction_divergence_signal']
        else:
            return row['transition_divergence_boost']
    
    data['adaptive_signal'] = data.apply(adaptive_signal_selection, axis=1)
    
    # Composite Alpha Construction
    data['expansion_core'] = data['expansion_divergence_signal'] * data['expansion_regime_score']
    data['contraction_core'] = data['contraction_divergence_signal'] * data['contraction_regime_score']
    data['transition_core'] = data['transition_divergence_boost'] * data['regime_transition_signal']
    
    def regime_weighted_divergence(row):
        if row['volatility_regime'] == 'Expansion':
            return row['expansion_core'] * row['expansion_divergence_multiplier']
        elif row['volatility_regime'] == 'Contraction':
            return row['contraction_core'] * row['contraction_divergence_multiplier']
        else:
            return row['transition_core'] * row['transition_divergence_multiplier']
    
    data['regime_weighted_divergence'] = data.apply(regime_weighted_divergence, axis=1)
    
    data['timeframe_enhanced_divergence'] = data['regime_weighted_divergence'] * data['integrated_regime_signal']
    data['volume_validation'] = data['timeframe_enhanced_divergence'] * data['volume_regime_correlation']
    
    # Final Alpha Factor
    data['divergence_base'] = data['volume_validation'] * data['timeframe_synchronization']
    data['quality_enhanced_output'] = data['divergence_base'] * data['regime_quality_metric']
    data['final_alpha'] = data['quality_enhanced_output'] * data['divergence_quality']
    
    return data['final_alpha']
