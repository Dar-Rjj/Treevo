import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Quantum Microstructure
    # High-Frequency Quantum Microstructure
    data['hf_quantum'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6)) * \
                        (data['volume'] / data['volume'].shift(1)) * \
                        (abs(data['close'] - data['close'].shift(2)) / 
                         (abs(data['close'] - data['close'].shift(1)) + abs(data['close'].shift(1) - data['close'].shift(2)) + 1e-6))
    
    # Medium-Frequency Quantum Microstructure
    data['mf_quantum'] = ((data['close'] - data['close'].shift(5)) / 
                         (abs(data['close'] - data['close'].shift(1)) + 
                          abs(data['close'].shift(1) - data['close'].shift(2)) +
                          abs(data['close'].shift(2) - data['close'].shift(3)) +
                          abs(data['close'].shift(3) - data['close'].shift(4)) +
                          abs(data['close'].shift(4) - data['close'].shift(5)) + 1e-6)) * \
                        ((data['high'] - data['low']) / (abs(data['open'] - data['close'].shift(1)) + 1e-6)) * \
                        (abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-6))
    
    # Low-Frequency Quantum Microstructure
    def calc_low_freq_quantum(data):
        result = []
        for i in range(len(data)):
            if i < 20:
                result.append(np.nan)
                continue
            close_range_sum = sum(abs(data['close'].iloc[j] - data['close'].iloc[j-1]) for j in range(i-19, i+1))
            lf_quantum = ((data['close'].iloc[i] - data['close'].iloc[i-20]) / (close_range_sum + 1e-6)) * \
                        (abs(data['open'].iloc[i] - data['close'].iloc[i-1]) / (abs(data['close'].iloc[i] - data['close'].iloc[i-20]) + 1e-6)) * \
                        (abs(data['open'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i] - data['low'].iloc[i] + 1e-6))
            result.append(lf_quantum)
        return result
    
    data['lf_quantum'] = calc_low_freq_quantum(data)
    
    # Quantum Microstructure Efficiency Patterns
    # Quantum Bid-Side Microstructure
    data['quantum_bid'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-6)) * \
                         (data['volume'] / (data['amount'] + 1e-6)) * \
                         ((data['close'] - data['low']) / (data['open'] - data['low'] + 1e-6))
    
    # Quantum Ask-Side Microstructure
    data['quantum_ask'] = ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-6)) * \
                         (data['volume'] / (data['amount'] + 1e-6)) * \
                         ((data['high'] - data['close']) / (data['high'] - data['open'] + 1e-6))
    
    # Quantum Microstructure Efficiency Differential
    data['quantum_eff_diff'] = (data['quantum_bid'] - data['quantum_ask']) * \
                              np.sign(data['close'] - data['open']) * \
                              ((data['amount'] / data['amount'].shift(1)) - 1)
    
    # Quantum Volume-Price Microstructure Dynamics
    # Quantum Volume Microstructure Gradient
    data['volume_gradient'] = (data['volume'] / data['volume'].shift(1)) * \
                             ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-6)) * \
                             np.sign(data['close'] - data['close'].shift(1)) * \
                             np.sign(data['volume'] - data['volume'].shift(1))
    
    # Quantum Price Microstructure Strength
    data['price_strength'] = (abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6)) * \
                            data['volume'] * \
                            ((data['close'] - data['close'].shift(1)) / (abs(data['close'] - data['close'].shift(1)) + 1e-6))
    
    # Quantum Volume-Price Microstructure Efficiency
    data['volume_price_eff'] = ((data['close'] - data['open']) * data['volume'] / (data['high'] - data['low'] + 1e-6)) * \
                              ((data['amount'] / data['volume']) / (data['amount'].shift(1) / data['volume'].shift(1)) - 1)
    
    # Quantum Microstructure Range Components
    # Quantum Range Utilization Efficiency
    data['range_util_eff'] = (abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6)) * \
                            data['volume'] * \
                            (abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6))
    
    # Quantum Microstructure Range Pressure
    data['range_pressure'] = ((data['close'] - data['low']) * (data['high'] - data['close']) / 
                             ((data['high'] - data['low'] + 1e-6) ** 2)) * \
                            data['volume'] * \
                            (data['close'] - (data['high'] + data['low']) / 2)
    
    # Quantum Range Asymmetry Momentum
    data['range_asymmetry'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / \
                             (data['high'] - data['low'] + 1e-6) * data['volume']
    
    # Quantum Microstructure Regime Classification
    # Quantum High-Frequency Momentum
    data['hf_momentum_regime'] = ((data['hf_quantum'] > 0.6) & 
                                 (data['volume_price_eff'] > 0.5) & 
                                 (data['volume'] / ((data['volume'].shift(2) + data['volume'].shift(1) + data['volume']) / 3) > 1)).astype(int)
    
    # Quantum Medium-Frequency Trend
    data['mf_trend_regime'] = ((data['mf_quantum'] > 0.5) & 
                              (np.sign(data['close'] - data['open']) * np.sign(data['close'].shift(1) - data['open'].shift(1)) > 0) & 
                              (abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6) > 0.3)).astype(int)
    
    # Quantum Low-Frequency Reversal
    data['lf_reversal_regime'] = ((data['lf_quantum'] < -0.3) & 
                                 (data['quantum_eff_diff'] < -0.2) & 
                                 ((data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5) + 1e-6) < 0.8)).astype(int)
    
    # Final Quantum Microstructure Alpha Calculation
    # Multi-Scale Quantum Component
    data['multi_scale_component'] = data['hf_quantum'] * 0.3 + data['mf_quantum'] * 0.35 + data['lf_quantum'] * 0.35
    
    # Quantum Efficiency Component
    data['efficiency_component'] = data['quantum_bid'] * 0.3 + data['quantum_ask'] * 0.3 + data['quantum_eff_diff'] * 0.4
    
    # Quantum Volume Microstructure Component
    data['volume_micro_component'] = data['volume_gradient'] * data['price_strength'] * 0.1 + 1
    
    # Quantum Range Component
    data['range_component'] = data['range_asymmetry'] * data['range_util_eff']
    
    # Base Quantum Microstructure Momentum
    data['base_momentum'] = data['hf_quantum'] * data['volume_price_eff'] * data['range_asymmetry']
    
    # Quantum Microstructure Regime Multiplier
    data['regime_multiplier'] = 1.0
    data.loc[data['hf_momentum_regime'] == 1, 'regime_multiplier'] = 1.4
    data.loc[data['mf_trend_regime'] == 1, 'regime_multiplier'] = 1.2
    data.loc[data['lf_reversal_regime'] == 1, 'regime_multiplier'] = 0.8
    
    # Quantum Divergence Adjustment
    data['divergence_adjustment'] = 0
    data.loc[data['range_pressure'] > 0.5, 'divergence_adjustment'] += 0.2
    data.loc[data['quantum_eff_diff'] < -0.5, 'divergence_adjustment'] -= 0.1
    
    # Final Quantum Microstructure Alpha
    data['quantum_alpha'] = ((data['base_momentum'] * data['efficiency_component'] * data['range_component']) * 
                            data['volume_micro_component'] * data['regime_multiplier'] + 
                            data['divergence_adjustment'])
    
    return data['quantum_alpha']
