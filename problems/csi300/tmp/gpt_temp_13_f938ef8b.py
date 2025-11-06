import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Fractal Dynamics
    # Fractal Flow Detection
    data['price_flow_fractal'] = (np.log(data['high'] - data['low'] + 1e-8) / 
                                 np.log(data['volume'] + 1e-8)) * np.log(data['amount'] + 1)
    
    data['volume_flow_fractal'] = (np.log(data['volume'] / data['volume'].shift(1) + 1e-8) / 
                                  np.log(data['amount'] / data['amount'].shift(1) + 1e-8)) * (data['volume'] / data['amount'])
    
    data['cross_flow_fractal'] = data['price_flow_fractal'] * data['volume_flow_fractal']
    
    # Fractal Momentum Enhancement
    data['fractal_momentum_enhancement'] = ((data['close'] - data['close'].shift(2)) / 
                                           (data['close'].shift(2) - data['close'].shift(4) + 1e-8)) * (data['volume'] / data['volume'].shift(1))
    
    # Volume Fractal Persistence
    vol_persistence = []
    for i in range(len(data)):
        if i < 3:
            vol_persistence.append(0)
            continue
        count = 0
        for j in range(1, 4):
            if (data['close'].iloc[i] > data['close'].iloc[i-j]) and (data['volume'].iloc[i] > data['volume'].iloc[i-j]):
                count += 1
        vol_persistence.append(count / 3)
    data['volume_fractal_persistence'] = vol_persistence
    
    # Asymmetric Compression Dynamics
    # Fractal Flow Compression
    data['flow_congestion'] = (data['high'] - data['low']) < (data['high'].shift(1) - data['low'].shift(1)) * 0.7
    
    data['fractal_compression_ratio'] = (data['high'] - data['low']) / (
        ((data['high'].shift(1) - data['low'].shift(1)) + 
         (data['high'].shift(2) - data['low'].shift(2)) + 
         (data['high'].shift(3) - data['low'].shift(3))) / 3 + 1e-8)
    
    # Fractal Flow Duration
    flow_duration = []
    consecutive_count = 0
    for i in range(len(data)):
        if i < 1:
            flow_duration.append(0)
            continue
        if data['flow_congestion'].iloc[i]:
            consecutive_count += 1
        else:
            consecutive_count = 0
        flow_duration.append(data['fractal_compression_ratio'].iloc[i] * consecutive_count)
    data['fractal_flow_duration'] = flow_duration
    
    # Asymmetric Range Bias
    data['gap_resistance_efficiency'] = ((data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)) * (
        (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8))
    
    data['asymmetric_range_bias'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * (
        (data['high'] - data['close']) / (data['close'] - data['low'] + 1e-8))
    
    data['compression_breakout_signal'] = (data['gap_resistance_efficiency'] * 
                                          data['fractal_compression_ratio'] * 
                                          data['asymmetric_range_bias'])
    
    # Fractal Breakout Quality
    data['fractal_amount_breakout'] = (data['amount'] / data['amount'].shift(5)) * np.sign(data['close'] - data['close'].shift(1)) * (data['volume'] / data['amount'])
    
    # Flow Price Breakout
    flow_price_breakout = []
    for i in range(len(data)):
        if i < 3:
            flow_price_breakout.append(0)
            continue
        max_high = max(data['high'].iloc[i-3:i])
        flow_price_breakout.append((data['close'].iloc[i] > max_high) * (data['amount'].iloc[i] / data['amount'].iloc[i-1] - 1))
    data['flow_price_breakout'] = flow_price_breakout
    
    # Fractal Breakout Quality
    fractal_breakout_quality = []
    for i in range(len(data)):
        if i < 3:
            fractal_breakout_quality.append(0)
            continue
        max_high = max(data['high'].iloc[i-3:i])
        fractal_breakout_quality.append((data['close'].iloc[i] - max_high) * 
                                       data['flow_price_breakout'].iloc[i] * 
                                       data['fractal_amount_breakout'].iloc[i])
    data['fractal_breakout_quality'] = fractal_breakout_quality
    
    # Flow-Pressure Asymmetry
    # Fractal Flow Pressure
    data['flow_up_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['amount'] * np.exp(-np.abs((data['close'] - data['close'].shift(1)) / data['close'].shift(1)))
    
    data['fractal_down_pressure'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * data['price_flow_fractal']
    
    data['flow_fractal_pressure_entropy'] = (data['flow_up_pressure'] / (data['flow_up_pressure'] + data['fractal_down_pressure'] + 1e-8)) * data['volume_fractal_persistence']
    
    # Microstructure Pressure Asymmetry
    data['fracture_high_side_pressure'] = ((data['high'] - data['close']) / (data['close'] - data['low'] + 1e-8)) * data['cross_flow_fractal'] * data['volume']
    
    data['fracture_low_side_support'] = ((data['close'] - data['low']) / (data['high'] - data['close'] + 1e-8)) * data['cross_flow_fractal'] * data['volume']
    
    data['pressure_efficiency'] = ((data['high'] - data['open']) / (data['open'] - data['low'] + 1e-8)) * (
        (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8))
    
    # Fractal Flow Momentum Divergence
    data['flow_ultra_short_momentum'] = ((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) * (data['amount'] / data['amount'].shift(1) - 1)
    
    data['fractal_short_term_momentum'] = (data['volume'] / data['volume'].shift(1)) * data['volume_flow_fractal']
    
    data['flow_fractal_momentum_divergence'] = data['flow_ultra_short_momentum'] - data['fractal_short_term_momentum']
    
    # Adaptive Fractal-Flow Synthesis
    # Core Fractal Components
    data['enhanced_fractal_momentum'] = data['fractal_momentum_enhancement'] * data['volume_fractal_persistence']
    
    data['pressure_flow_divergence'] = (data['fracture_high_side_pressure'] - data['fracture_low_side_support']) * data['flow_fractal_momentum_divergence']
    
    data['gap_compression_dynamics'] = data['gap_resistance_efficiency'] * data['compression_breakout_signal']
    
    # Fractal Flow Efficiency
    data['flow_range_efficiency'] = (np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)) * (data['close'] - data['open']) * data['amount']
    
    data['fractal_amount_efficiency'] = (data['amount'] / (data['volume'] / (data['high'] - data['low'] + 1e-8))) * np.log(data['amount'] + 1)
    
    data['cross_flow_efficiency'] = data['flow_range_efficiency'] * data['fractal_amount_efficiency']
    
    # Adaptive Alpha Generation
    data['base_momentum_alpha'] = data['enhanced_fractal_momentum'] * data['pressure_flow_divergence']
    
    data['flow_efficiency_enhancement'] = data['base_momentum_alpha'] * data['cross_flow_efficiency']
    
    data['compression_breakout_alignment'] = data['flow_efficiency_enhancement'] * data['compression_breakout_signal']
    
    # Multi-Regime Fractal Integration
    # Fractal Flow Regime Classification
    data['high_fractal_flow'] = (data['cross_flow_fractal'] > data['volume_fractal_persistence']).astype(float)
    data['low_fractal_flow'] = (data['cross_flow_fractal'] < data['volume_fractal_persistence'] * 0.3).astype(float)
    data['fractal_flow_transition'] = (np.abs(data['cross_flow_fractal'] - data['volume_fractal_persistence']) > 
                                      np.abs(data['volume'] / data['volume'].shift(1) - 1)).astype(float)
    
    # Regime-Specific Momentum
    data['high_fractal_momentum'] = data['flow_fractal_momentum_divergence'] * data['high_fractal_flow']
    data['low_fractal_momentum'] = data['flow_ultra_short_momentum'] * data['low_fractal_flow']
    data['flow_transition_momentum'] = data['flow_fractal_pressure_entropy'] * data['fractal_flow_transition']
    
    # Asymmetric Regime Patterns
    data['high_flow_asymmetry'] = data['fracture_high_side_pressure'] * data['high_fractal_flow']
    data['low_flow_support'] = data['fracture_low_side_support'] * data['low_fractal_flow']
    data['transition_pressure'] = data['pressure_efficiency'] * data['fractal_flow_transition']
    
    # Final Asymmetric Fractal-Flow Alpha
    data['primary_fractal_factor'] = (data['high_fractal_momentum'] * data['high_flow_asymmetry'] + 
                                     data['low_fractal_momentum'] * data['low_flow_support'] + 
                                     data['flow_transition_momentum'] * data['transition_pressure'])
    
    data['asymmetric_breakout_enhancement'] = (data['primary_fractal_factor'] * 
                                              data['fractal_breakout_quality'] * 
                                              data['compression_breakout_signal'])
    
    data['final_asymmetric_fractal_flow_alpha'] = (data['asymmetric_breakout_enhancement'] * 
                                                  data['cross_flow_fractal'] * 
                                                  data['flow_fractal_pressure_entropy'])
    
    return data['final_asymmetric_fractal_flow_alpha']
