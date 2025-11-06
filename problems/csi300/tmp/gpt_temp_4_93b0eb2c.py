import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-scale Price Momentum
    data['ultra_short_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['short_term_momentum'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Medium-term Momentum denominator calculation
    range_sum = (data['high'] - data['low']) + \
                (data['high'].shift(1) - data['low'].shift(1)) + \
                (data['high'].shift(2) - data['low'].shift(2)) + \
                (data['high'].shift(3) - data['low'].shift(3)) + \
                (data['high'].shift(4) - data['low'].shift(4))
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(4)) / range_sum
    
    data['momentum_fractal_score'] = data['ultra_short_momentum'] * data['short_term_momentum'] * data['medium_term_momentum']
    
    # Momentum Acceleration Patterns
    data['momentum_change_rate'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(2))
    
    # Momentum Persistence
    momentum_signs = pd.DataFrame()
    for i in range(4, -1, -1):
        momentum_signs[f'sign_{i}'] = np.sign(data['close'].shift(i) - data['close'].shift(i+1))
    
    momentum_persistence = []
    for idx in range(len(data)):
        if idx < 5:
            momentum_persistence.append(0)
            continue
        count = 0
        for i in range(4):
            if (momentum_signs.iloc[idx, i] == momentum_signs.iloc[idx, i+1]) and \
               (momentum_signs.iloc[idx, i] != 0):
                count += 1
        momentum_persistence.append(count)
    
    data['momentum_persistence'] = momentum_persistence
    
    # Momentum Volatility
    abs_returns = abs(data['close'] - data['close'].shift(1))
    data['momentum_volatility'] = abs_returns / ((abs_returns.shift(4) + abs_returns.shift(3) + 
                                                abs_returns.shift(2) + abs_returns.shift(1) + abs_returns) / 5)
    
    data['acceleration_composite'] = data['momentum_change_rate'] * data['momentum_persistence'] * data['momentum_volatility']
    
    # Fractal Momentum Integration
    data['scale_convergence'] = data['momentum_fractal_score'] * data['acceleration_composite']
    data['momentum_stability'] = (1 / abs(data['momentum_change_rate'])) * data['momentum_persistence']
    data['fractal_efficiency'] = data['scale_convergence'] / data['momentum_volatility']
    data['fractal_momentum_core'] = data['scale_convergence'] * data['momentum_stability'] * data['fractal_efficiency']
    
    # Volume Acceleration Dynamics
    data['volume_change_rate'] = data['volume'] / data['volume'].shift(1)
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Volume Persistence
    volume_increases = pd.DataFrame()
    for i in range(4, -1, -1):
        volume_increases[f'inc_{i}'] = (data['volume'].shift(i) > data['volume'].shift(i+1)).astype(int)
    
    volume_persistence = []
    for idx in range(len(data)):
        if idx < 5:
            volume_persistence.append(0)
            continue
        count = volume_increases.iloc[idx].sum()
        volume_persistence.append(count)
    
    data['volume_persistence'] = volume_persistence
    data['volume_momentum_score'] = data['volume_change_rate'] * data['volume_acceleration'] * data['volume_persistence']
    
    # Volume-Price Acceleration Alignment
    data['momentum_volume_correlation'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['acceleration_synchronization'] = data['momentum_change_rate'] * data['volume_acceleration']
    data['volume_momentum_efficiency'] = abs(data['close'] - data['close'].shift(1)) * data['volume'] / (data['high'] - data['low'])
    data['alignment_composite'] = data['momentum_volume_correlation'] * data['acceleration_synchronization'] * data['volume_momentum_efficiency']
    
    # Volume Acceleration Integration
    data['volume_momentum_core'] = data['volume_momentum_score'] * data['alignment_composite']
    data['acceleration_consistency'] = data['volume_persistence'] * data['momentum_persistence']
    data['volume_price_coherence'] = data['volume_momentum_core'] / data['momentum_volatility']
    data['volume_acceleration_factor'] = data['volume_momentum_core'] * data['acceleration_consistency'] * data['volume_price_coherence']
    
    # Fractal Range Expansion
    data['range_momentum'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['range_acceleration'] = data['range_momentum'] - ((data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(2) - data['low'].shift(2)))
    
    # Range Persistence
    range_increases = pd.DataFrame()
    for i in range(4, -1, -1):
        range_increases[f'inc_{i}'] = ((data['high'].shift(i) - data['low'].shift(i)) > 
                                     (data['high'].shift(i+1) - data['low'].shift(i+1))).astype(int)
    
    range_persistence = []
    for idx in range(len(data)):
        if idx < 5:
            range_persistence.append(0)
            continue
        count = range_increases.iloc[idx].sum()
        range_persistence.append(count)
    
    data['range_persistence'] = range_persistence
    data['range_expansion_score'] = data['range_momentum'] * data['range_acceleration'] * data['range_persistence']
    
    # Range-Price Fractal Alignment
    data['range_momentum_correlation'] = np.sign((data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1))) * \
                                       np.sign(data['close'] - data['close'].shift(1))
    data['range_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['range_price_synchronization'] = data['range_momentum'] * data['medium_term_momentum']
    data['range_alignment_composite'] = data['range_momentum_correlation'] * data['range_efficiency'] * data['range_price_synchronization']
    
    # Fractal Range Integration
    data['range_expansion_core'] = data['range_expansion_score'] * data['range_alignment_composite']
    data['range_stability'] = (1 / abs(data['range_acceleration'])) * data['range_persistence']
    data['range_price_coherence'] = data['range_expansion_core'] / data['momentum_volatility']
    data['fractal_range_factor'] = data['range_expansion_core'] * data['range_stability'] * data['range_price_coherence']
    
    # Dynamic Fractal Integration
    data['momentum_volume_fractal'] = data['fractal_momentum_core'] * data['volume_acceleration_factor']
    data['momentum_range_fractal'] = data['fractal_momentum_core'] * data['fractal_range_factor']
    data['volume_range_fractal'] = data['volume_acceleration_factor'] * data['fractal_range_factor']
    data['fractal_integration_core'] = data['momentum_volume_fractal'] * data['momentum_range_fractal'] * data['volume_range_fractal']
    
    # Fractal Timeframe Synthesis
    data['ultra_short_component'] = data['ultra_short_momentum'] * data['volume_change_rate'] * data['range_momentum']
    data['short_term_component'] = data['short_term_momentum'] * data['volume_acceleration'] * data['range_acceleration']
    data['medium_term_component'] = data['medium_term_momentum'] * data['volume_persistence'] * data['range_persistence']
    data['timeframe_fractal_signal'] = data['ultra_short_component'] * data['short_term_component'] * data['medium_term_component']
    
    # Fractal Weighting Mechanism
    data['momentum_consistency_weight'] = data['momentum_persistence'] * data['acceleration_consistency']
    data['volume_alignment_weight'] = data['momentum_volume_correlation'] * data['volume_price_coherence']
    data['range_synchronization_weight'] = data['range_momentum_correlation'] * data['range_price_coherence']
    data['fractal_weight_composite'] = data['momentum_consistency_weight'] * data['volume_alignment_weight'] * data['range_synchronization_weight']
    
    # Final Alpha Construction
    data['integrated_fractal_signal'] = data['fractal_integration_core'] * data['timeframe_fractal_signal']
    data['weighted_fractal_signal'] = data['integrated_fractal_signal'] * data['fractal_weight_composite']
    data['volatility_adjustment'] = data['weighted_fractal_signal'] / data['momentum_volatility']
    data['range_confirmation'] = data['volatility_adjustment'] * data['range_expansion_score']
    
    # Acceleration Refinement
    data['momentum_acceleration_final'] = data['range_confirmation'] * data['acceleration_composite']
    data['volume_acceleration_final'] = data['momentum_acceleration_final'] * data['volume_acceleration_factor']
    data['range_acceleration_final'] = data['volume_acceleration_final'] * data['range_acceleration']
    data['acceleration_alpha_core'] = data['range_acceleration_final'] * data['alignment_composite']
    
    # Fractal Alpha Output
    data['momentum_normalization'] = data['acceleration_alpha_core'] / data['momentum_fractal_score']
    data['volume_adjustment'] = data['momentum_normalization'] * data['volume_momentum_score']
    data['final_fractal_alpha'] = data['volume_adjustment'] * data['range_expansion_score']
    
    return data['final_fractal_alpha']
