import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Volatility Framework
    # Multi-Scale Volatility Fractals
    data['micro_vol_fractal'] = (data['high'] - data['low']) / data['close']
    data['short_vol_fractal'] = (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()) / data['close']
    data['medium_vol_fractal'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / data['close']
    
    # Volume-Volatility Fractal Dynamics
    data['volume_vol_intensity'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['amount_vol_concentration'] = (data['amount'] / data['volume']) * (data['high'] - data['low']) / data['close']
    data['vol_fractal_ratio'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))) * (data['volume'] / data['volume'].shift(1))
    
    # Fractal Volatility Regime Classification
    data['high_vol_regime'] = ((data['micro_vol_fractal'] > data['short_vol_fractal']) & 
                              (data['volume_vol_intensity'] > (data['volume'].shift(1) / data['volume'].shift(2))))
    data['low_vol_regime'] = ((data['micro_vol_fractal'] < data['short_vol_fractal']) & 
                             (data['volume_vol_intensity'] < (data['volume'].shift(1) / data['volume'].shift(2))))
    data['transition_vol_regime'] = (data['medium_vol_fractal'] > ((data['micro_vol_fractal'] + data['short_vol_fractal']) / 2))
    
    # Asymmetric Volatility-Volume Response
    # Directional Volatility Response Components
    price_range = data['close'] - data['open']
    data['upside_vol_response'] = (data['high'] - data['close']) / price_range.replace(0, np.nan)
    data['downside_vol_response'] = (data['close'] - data['low']) / price_range.replace(0, np.nan)
    data['vol_response_asymmetry'] = data['upside_vol_response'] / data['downside_vol_response'].replace(0, np.nan)
    
    # Volume-Volatility Response Framework
    data['volume_vol_momentum'] = (data['volume'] / data['volume'].shift(1)) * (price_range / (data['high'] - data['low']).replace(0, np.nan))
    data['vol_gap_response'] = (abs(data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)) * np.sign(data['close'] - data['open'])
    data['vol_close_response'] = (abs(data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)) * (data['volume'] / data['volume'].shift(1))
    
    # Volatility Response Persistence
    vol_response_persistence = []
    for i in range(len(data)):
        if i < 3:
            vol_response_persistence.append(0)
        else:
            count = sum((data['close'].iloc[j] - data['open'].iloc[j]) > (data['close'].iloc[j-1] - data['open'].iloc[j-1]) 
                       for j in range(i-2, i+1))
            vol_response_persistence.append(count * (data['volume'].iloc[i] / data['volume'].iloc[i-1]))
    data['vol_response_persistence'] = vol_response_persistence
    
    # Asymmetric Volatility Integration
    data['vol_response_divergence'] = data['vol_response_asymmetry'] * data['volume_vol_momentum']
    data['directional_vol_gap_asymmetry'] = data['vol_gap_response'] * data['upside_vol_response'] / data['downside_vol_response'].replace(0, np.nan)
    data['persistent_vol_asymmetry'] = data['vol_response_persistence'] * data['vol_response_asymmetry']
    
    # Fractal-Driven Volatility Momentum Divergence
    # Volatility Fractal Divergence Components
    data['price_volume_vol_fractal'] = (price_range / data['volume']) * (data['volume'] / data['volume'].shift(1))
    data['amount_vol_fractal_efficiency'] = (data['amount'] / data['volume']) * (price_range / (data['high'] - data['low']).replace(0, np.nan))
    
    vol_flow_prev = (data['close'].shift(1) - data['open'].shift(1)) / data['volume'].shift(1)
    data['volume_price_vol_flow'] = (price_range / data['volume']) - vol_flow_prev * (data['volume'] / data['volume'].shift(1))
    
    data['vol_fractal_momentum'] = ((data['volume'] / data['amount']) / (data['volume'].shift(3) / data['amount'].shift(3))) * (price_range / (data['close'].shift(1) - data['open'].shift(1)).replace(0, np.nan))
    
    # Volatility Reversal-Fractal Framework
    data['volume_vol_fractal_signal'] = np.where(data['volume'] < data['volume'].shift(1), 
                                                (data['volume'] / data['volume'].shift(1)) * (price_range / (data['high'] - data['low']).replace(0, np.nan)), 0)
    data['vol_fractal_momentum_signal'] = (data['volume'] / data['volume'].shift(1)) * price_range
    data['vol_transition_fractal'] = price_range * np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(data['volume'].shift(1) - data['volume'].shift(2))
    
    # Dynamic Volatility Fractal Selection
    data['vol_fractal_regime'] = np.where(data['volume'] > data['volume'].shift(1) * 1.1, 'high',
                                         np.where(data['volume'] < data['volume'].shift(1) * 0.9, 'low', 'transition'))
    
    data['high_vol_fractal_divergence'] = data['vol_fractal_momentum_signal'] * data['price_volume_vol_fractal']
    data['low_vol_fractal_divergence'] = data['volume_vol_fractal_signal'] * data['amount_vol_fractal_efficiency']
    data['transition_vol_fractal_divergence'] = data['vol_transition_fractal'] * data['volume_price_vol_flow']
    
    # Multi-Scale Volatility Fractal Convergence
    # Volatility-Volume Fractal Alignment
    data['vol_fractal_correlation'] = (np.sign(data['micro_vol_fractal'] - data['micro_vol_fractal'].shift(1)) * 
                                     np.sign(data['volume_vol_intensity'] - (data['volume'].shift(1) / data['volume'].shift(2))))
    data['vol_fractal_consistency'] = data['medium_vol_fractal'] * data['amount_vol_concentration']
    
    regime_strength = np.where(data['high_vol_regime'], 1.0, np.where(data['low_vol_regime'], -1.0, 0.0))
    data['vol_regime_fractal_sync'] = regime_strength * data['vol_fractal_ratio']
    
    # Multi-Scale Volatility Fractal Integration
    data['vol_fractal_acceleration'] = data['volume_vol_momentum'] - data['volume_vol_momentum'].shift(1)
    data['volume_vol_concentration'] = (data['volume'] / data['amount'] - data['volume'].shift(1) / data['amount'].shift(1)) * (data['volume'] / data['volume'].shift(2))
    
    # Volatility Fractal Compression
    avg_prev_range = (data['high'].shift(1) - data['low'].shift(1) + 
                     data['high'].shift(2) - data['low'].shift(2) + 
                     data['high'].shift(3) - data['low'].shift(3)) / 3
    data['vol_fractal_compression'] = ((data['high'] - data['low']) / avg_prev_range.replace(0, np.nan)) * (data['volume'] / data['volume'].shift(1))
    
    data['volume_vol_fractal'] = (data['volume'] / data['volume'].shift(1)) * ((data['amount'] / data['volume']) / (data['amount'].shift(1) / data['volume'].shift(1)))
    
    # Integrated Volatility Fractal Synthesis
    data['core_vol_fractal'] = data['vol_response_divergence'] * data['amount_vol_concentration']
    
    short_term = data['vol_fractal_acceleration'] * data['volume_vol_concentration']
    medium_term = data['vol_response_persistence'] * data['vol_fractal_momentum']
    long_term = data['vol_fractal_compression'] * data['volume_vol_fractal']
    
    data['multi_scale_vol_enhancement'] = data['core_vol_fractal'] * short_term * medium_term
    data['vol_fractal_enhanced_output'] = data['multi_scale_vol_enhancement'] * (1 + abs(data['vol_fractal_acceleration'])) * (1 + data['vol_response_persistence'])
    
    # Structural Volatility Fractal Break
    # Volatility Fractal Break Framework
    data['price_vol_fractal_break'] = abs(price_range) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    data['volume_vol_fractal_break'] = (data['volume'] / data['volume'].shift(1)) * (data['volume'].shift(1) / data['volume'].shift(2))
    data['vol_fractal_break_signal'] = data['price_vol_fractal_break'] * data['volume_vol_fractal_break']
    
    # Volatility Break-Fractal Interaction
    data['vol_break_response_alignment'] = (np.sign(data['vol_fractal_break_signal'] - data['vol_fractal_break_signal'].shift(1)) * 
                                          np.sign(data['vol_response_asymmetry'] - 1))
    data['vol_break_volume_coupling'] = data['vol_fractal_break_signal'] * data['volume_vol_intensity']
    data['vol_break_fractal_integration'] = data['vol_fractal_break_signal'] * data['volume_vol_momentum']
    
    # Volatility Break-Enhanced Fractals
    data['vol_response_break_enhancement'] = data['persistent_vol_asymmetry'] * data['vol_fractal_break_signal']
    data['vol_volume_break_amplification'] = data['amount_vol_concentration'] * data['vol_fractal_break_signal']
    data['vol_gap_break_integration'] = data['directional_vol_gap_asymmetry'] * data['vol_fractal_break_signal']
    
    # Composite Volatility Fractal Alpha
    # Volatility Regime-Weighted Fractal Response
    data['high_vol_allocation'] = data['vol_response_break_enhancement'] * data['high_vol_fractal_divergence']
    data['low_vol_allocation'] = data['vol_volume_break_amplification'] * data['low_vol_fractal_divergence']
    data['transition_vol_allocation'] = data['vol_gap_break_integration'] * data['transition_vol_fractal_divergence']
    
    # Multi-Scale Volatility Fractal Integration
    data['core_vol_fractal_alignment'] = data['core_vol_fractal'] * data['vol_fractal_consistency']
    data['vol_break_enhanced_core'] = data['core_vol_fractal_alignment'] * data['vol_break_response_alignment']
    data['vol_response_refined_core'] = data['vol_break_enhanced_core'] * data['vol_response_divergence']
    
    # Dynamic Volatility Fractal Adjustment
    data['vol_fractal_specific_weighting'] = np.where(data['high_vol_regime'], data['high_vol_allocation'],
                                                     np.where(data['low_vol_regime'], data['low_vol_allocation'], 
                                                             data['transition_vol_allocation']))
    
    data['volume_enhanced_vol_response'] = data['vol_fractal_specific_weighting'] * data['vol_response_refined_core']
    data['vol_fractal_confirmed_output'] = data['volume_enhanced_vol_response'] * data['vol_fractal_compression']
    
    # Final Volatility Fractal Alpha
    data['core_vol_alpha'] = data['vol_fractal_confirmed_output'] * data['vol_fractal_enhanced_output']
    
    # Volatility Quality Enhancement
    vol_quality = (data['volume_vol_intensity'].rolling(window=5).std() / data['volume_vol_intensity'].rolling(window=5).mean()).replace(np.inf, 1).replace(-np.inf, 1)
    data['vol_quality_enhanced'] = data['core_vol_alpha'] * (1 / (1 + abs(vol_quality)))
    
    # Volatility Fractal Confidence
    vol_stability = (data['micro_vol_fractal'].rolling(window=3).std() / data['micro_vol_fractal'].rolling(window=3).mean()).replace(np.inf, 1).replace(-np.inf, 1)
    data['final_vol_fractal_alpha'] = data['vol_quality_enhanced'] * (1 / (1 + abs(vol_stability)))
    
    # Return the final alpha factor
    return data['final_vol_fractal_alpha']
