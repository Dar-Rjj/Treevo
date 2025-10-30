import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Fractal Momentum Structure
    # Multi-Scale Momentum Decay
    data['short_term_momentum_decay'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(3) - data['close'].shift(6))
    
    # Medium-term momentum persistence
    close_gt_prev = (data['close'] > data['close'].shift(1)).astype(int)
    data['medium_term_persistence'] = close_gt_prev.rolling(window=8, min_periods=1).sum() / 8
    
    data['momentum_decay_ratio'] = data['short_term_momentum_decay'] / data['medium_term_persistence']
    data['momentum_decay_ratio'] = data['momentum_decay_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Fractal Range Analysis
    data['micro_range_efficiency'] = ((data['close'] - data['low']) / (data['high'] - data['low'])) - \
                                   ((data['close'].shift(1) - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)))
    
    data['macro_range_alignment'] = ((data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))) * \
                                  ((data['close'] - data['open']) / (data['high'] - data['low']))
    
    # Fractal range consistency (correlation over 5 days)
    micro_series = data['micro_range_efficiency']
    macro_series = data['macro_range_alignment']
    fractal_consistency = []
    for i in range(len(data)):
        if i < 4:
            fractal_consistency.append(np.nan)
        else:
            window_micro = micro_series.iloc[i-4:i+1]
            window_macro = macro_series.iloc[i-4:i+1]
            if len(window_micro.dropna()) >= 3 and len(window_macro.dropna()) >= 3:
                corr = window_micro.corr(window_macro)
                fractal_consistency.append(corr if not np.isnan(corr) else 0)
            else:
                fractal_consistency.append(0)
    data['fractal_range_consistency'] = fractal_consistency
    
    # Momentum-Fractal Integration
    data['decay_adjusted_efficiency'] = data['momentum_decay_ratio'] * data['fractal_range_consistency']
    data['range_momentum_synchronization'] = data['macro_range_alignment'] * data['short_term_momentum_decay']
    data['fractal_momentum_quality'] = data['decay_adjusted_efficiency'] * data['range_momentum_synchronization']
    
    # Volume Synchronization Framework
    # Volume-Flow Dynamics
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    
    volume_gt_prev = (data['volume'] > data['volume'].shift(1)).astype(int)
    data['volume_persistence'] = volume_gt_prev.rolling(window=5, min_periods=1).sum() / 5
    
    data['volume_flow_momentum'] = data['volume_acceleration'] * data['volume_persistence']
    
    # Price-Volume Fractal Alignment
    data['volume_range_synchronization'] = (data['volume'] / (data['high'] - data['low'])) * data['volume_flow_momentum']
    data['price_volume_fractal'] = ((data['close'] - data['open']) * data['volume']) / \
                                 ((data['close'].shift(1) - data['open'].shift(1)) * data['volume'].shift(1))
    data['price_volume_fractal'] = data['price_volume_fractal'].replace([np.inf, -np.inf], np.nan)
    
    data['fractal_volume_efficiency'] = data['volume_range_synchronization'] * data['price_volume_fractal']
    
    # Synchronization Quality Assessment
    data['volume_momentum_alignment'] = data['volume_flow_momentum'] * data['short_term_momentum_decay']
    data['range_volume_consistency'] = data['fractal_volume_efficiency'] * data['macro_range_alignment']
    data['synchronization_strength'] = data['volume_momentum_alignment'] * data['range_volume_consistency']
    
    # Efficiency Pattern Recognition
    # Intraday Pattern Analysis
    data['opening_efficiency_momentum'] = ((data['open'] - data['low']) / (data['high'] - data['low'])) - \
                                        ((data['open'].shift(1) - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)))
    
    data['closing_efficiency_quality'] = ((data['close'] - data['low']) / (data['high'] - data['low'])) * \
                                       ((data['close'] / data['open']) - 1)
    
    closing_quality_gt_zero = (data['closing_efficiency_quality'] > 0).astype(int)
    data['intraday_pattern_persistence'] = closing_quality_gt_zero.rolling(window=3, min_periods=1).sum() / 3
    
    # Multi-Timeframe Efficiency
    data['short_term_efficiency_trend'] = data['closing_efficiency_quality'] - data['closing_efficiency_quality'].shift(2)
    data['medium_term_efficiency_stability'] = data['closing_efficiency_quality'].rolling(window=8, min_periods=1).std()
    data['efficiency_timeframe_alignment'] = data['short_term_efficiency_trend'] / data['medium_term_efficiency_stability']
    data['efficiency_timeframe_alignment'] = data['efficiency_timeframe_alignment'].replace([np.inf, -np.inf], np.nan)
    
    # Pattern-Volume Integration
    data['volume_weighted_efficiency'] = data['closing_efficiency_quality'] * data['volume']
    data['pattern_volume_synchronization'] = data['intraday_pattern_persistence'] * data['volume_persistence']
    data['efficiency_volume_quality'] = data['volume_weighted_efficiency'] * data['pattern_volume_synchronization']
    
    # Decay-Synchronization Validation
    # Momentum-Decay Confirmation
    data['decay_volume_alignment'] = data['momentum_decay_ratio'] * data['volume_flow_momentum']
    data['fractal_decay_consistency'] = data['fractal_momentum_quality'] * data['momentum_decay_ratio']
    data['decay_synchronization'] = data['decay_volume_alignment'] * data['fractal_decay_consistency']
    
    # Volume-Pattern Convergence
    data['volume_efficiency_convergence'] = data['efficiency_volume_quality'] * data['volume_momentum_alignment']
    data['pattern_synchronization_alignment'] = data['pattern_volume_synchronization'] * data['synchronization_strength']
    data['convergence_quality'] = data['volume_efficiency_convergence'] * data['pattern_synchronization_alignment']
    
    # Multi-Dimensional Validation
    data['fractal_validation'] = data['fractal_range_consistency'] * data['fractal_volume_efficiency']
    data['decay_pattern_integration'] = data['momentum_decay_ratio'] * data['intraday_pattern_persistence']
    
    sync_strength_gt_zero = (data['synchronization_strength'] > 0).astype(int)
    data['synchronization_persistence'] = sync_strength_gt_zero.rolling(window=3, min_periods=1).sum() / 3
    
    # Composite Alpha Generation
    # Core Momentum-Decay Factor
    data['volume_synchronized_decay'] = data['decay_synchronization'] * data['volume_flow_momentum']
    data['fractal_decay_momentum'] = data['fractal_momentum_quality'] * data['momentum_decay_ratio']
    data['core_decay_factor'] = data['volume_synchronized_decay'] * data['fractal_decay_momentum']
    
    # Pattern-Synchronization Enhancement
    data['efficiency_synchronized_patterns'] = data['efficiency_volume_quality'] * data['pattern_synchronization_alignment']
    data['volume_pattern_convergence'] = data['pattern_volume_synchronization'] * data['volume_efficiency_convergence']
    data['enhanced_pattern_factor'] = data['efficiency_synchronized_patterns'] * data['volume_pattern_convergence']
    
    # Final Alpha Composite
    data['decay_pattern_integration_final'] = data['core_decay_factor'] * data['enhanced_pattern_factor']
    data['synchronization_validation'] = data['convergence_quality'] * data['synchronization_persistence']
    
    # Final alpha factor
    data['alpha_factor'] = data['decay_pattern_integration_final'] * data['synchronization_validation'] * data['fractal_validation']
    
    # Clean infinite values and return the final alpha factor
    data['alpha_factor'] = data['alpha_factor'].replace([np.inf, -np.inf], np.nan)
    
    return data['alpha_factor']
