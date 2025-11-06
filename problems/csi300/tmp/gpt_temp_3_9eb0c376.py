import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-scale Momentum Resonance
    data['high_freq_momentum'] = ((data['close'] / data['close'].shift(2) - 1) * 
                                 (data['high'] - data['low']) / 
                                 (data['high'].shift(1) - data['low'].shift(1)))
    
    data['medium_freq_momentum'] = ((data['close'] / data['close'].shift(5) - 1) * 
                                   (data['high'].shift(2) - data['low'].shift(2)) / 
                                   (data['high'].shift(4) - data['low'].shift(4)))
    
    data['low_freq_momentum'] = ((data['close'] / data['close'].shift(10) - 1) * 
                                (data['high'].shift(5) - data['low'].shift(5)) / 
                                (data['high'].shift(8) - data['low'].shift(8)))
    
    # Volatility Harmonic Resonance
    data['upper_volatility'] = ((data['high'] - data['close']) / 
                               (data['high'].shift(1) - data['low'].shift(1)) * 
                               (data['close'] / data['close'].shift(2) - 1))
    
    data['lower_volatility'] = ((data['close'] - data['low']) / 
                               (data['high'].shift(1) - data['low'].shift(1)) * 
                               (data['close'] / data['close'].shift(2) - 1))
    
    data['net_resonance_amplitude'] = data['lower_volatility'] - data['upper_volatility']
    
    # Volume-Frequency Dynamics
    data['volume_rhythm'] = ((data['volume'] / data['volume'].shift(1)) * 
                            (data['close'] - data['open']) / 
                            (data['high'] - data['low']))
    
    data['frequency_impact'] = (((data['high'] - data['low']) / data['volume']) * 
                               (data['close'] - data['open']) / 
                               (data['high'] - data['low']))
    
    data['volume_frequency'] = ((data['volume'] / (data['high'] - data['low'])) * 
                               (data['close'] - data['low']) / 
                               (data['high'] - data['low']))
    
    # Price-Volatility Wave Patterns
    data['wave_density'] = (((data['high'] - data['low']) / (data['close'] - data['open'])) * 
                           (data['high'] - data['low']) / 
                           (data['high'].shift(1) - data['low'].shift(1)))
    
    data['wave_impact'] = ((abs(data['close'] - data['open']) / (data['high'] - data['low'])) * 
                          (data['high'] - data['low']) / 
                          (data['close'] - data['open']))
    
    data['intraday_pressure'] = (((data['close'] - data['open']) / (data['high'] - data['low'])) * 
                                (data['high'] - data['close']) / 
                                (data['high'] - data['low']))
    
    # Breakout Detection
    # Calculate rolling max for resonance breakout
    rolling_resonance = []
    for i in range(len(data)):
        if i >= 15:
            window_data = data.iloc[i-15:i]
            values = ((window_data['close'].values[:-2] / window_data['close'].values[2:] - 1) * 
                     (window_data['high'].values[:-2] - window_data['low'].values[:-2]) / 
                     (window_data['high'].values[1:-1] - window_data['low'].values[1:-1]))
            rolling_resonance.append(np.max(values) if len(values) > 0 else 1)
        else:
            rolling_resonance.append(1)
    
    data['rolling_max_resonance'] = rolling_resonance
    data['resonance_breakout'] = data['high_freq_momentum'] / data['rolling_max_resonance']
    
    data['volume_weighted_resonance'] = data['high_freq_momentum'] * data['volume']
    
    data['regime_shift'] = ((data['high_freq_momentum'] + data['medium_freq_momentum']) / 
                           data['low_freq_momentum'])
    
    # Core Components
    data['resonance_component'] = (data['net_resonance_amplitude'] * 
                                  (data['high_freq_momentum'] - data['medium_freq_momentum']) * 
                                  data['wave_density'])
    
    data['frequency_component'] = (data['frequency_impact'] * 
                                  data['volume_frequency'] * 
                                  data['intraday_pressure'])
    
    data['breakout_component'] = data['volume_weighted_resonance'] * data['regime_shift']
    
    # Regime Weighting
    high_freq_condition = ((data['volume'] > data['volume'].shift(4)) & 
                          (data['high_freq_momentum'] > data['medium_freq_momentum']))
    
    low_freq_condition = ((data['volume'] <= data['volume'].shift(4)) & 
                         (data['high_freq_momentum'] <= data['medium_freq_momentum']))
    
    breakout_condition = ((data['resonance_breakout'] > 1.5) | 
                         (data['volume'] / data['volume'].shift(1) > 1.5))
    
    # Initialize weights
    weight_resonance = np.zeros(len(data))
    weight_frequency = np.zeros(len(data))
    weight_breakout = np.zeros(len(data))
    
    # Apply regime weights
    weight_resonance[high_freq_condition] = 0.4
    weight_frequency[high_freq_condition] = 0.3
    weight_breakout[high_freq_condition] = 0.3
    
    weight_resonance[low_freq_condition] = 0.25
    weight_frequency[low_freq_condition] = 0.35
    weight_breakout[low_freq_condition] = 0.4
    
    weight_resonance[breakout_condition] = 0.3
    weight_frequency[breakout_condition] = 0.25
    weight_breakout[breakout_condition] = 0.45
    
    # Default weights for remaining cases
    default_mask = ~(high_freq_condition | low_freq_condition | breakout_condition)
    weight_resonance[default_mask] = 0.33
    weight_frequency[default_mask] = 0.33
    weight_breakout[default_mask] = 0.34
    
    # Final Alpha
    data['final_alpha'] = ((data['resonance_component'] * weight_resonance + 
                           data['frequency_component'] * weight_frequency + 
                           data['breakout_component'] * weight_breakout) * 
                          data['volume_weighted_resonance'] * 
                          data['frequency_impact'])
    
    return data['final_alpha']
