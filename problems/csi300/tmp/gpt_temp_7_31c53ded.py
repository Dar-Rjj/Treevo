import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Quantum-Entropic Dynamics
    # Micro Quantum-Entropic
    micro_quantum_entropic = ((data['close'] - data['open']) * 
                             (data['volume'] - data['volume'].shift(1)) / 
                             (data['high'] - data['low']) * 
                             (data['high'] - data['low']) / 
                             (np.abs(data['close'] - data['open']) + 0.0001))
    
    # Meso Quantum-Entropic
    def calculate_meso(row_index):
        if row_index < 3:
            return np.nan
        window = data.iloc[row_index-3:row_index+1]
        numerator = ((window['high'] - window['low']) * (window['close'] - window['open'])).sum()
        denominator = window['close'].sum()
        return numerator / denominator * data.iloc[row_index]['volume'] if denominator != 0 else np.nan
    
    meso_quantum_entropic = pd.Series([calculate_meso(i) for i in range(len(data))], index=data.index)
    
    # Macro Quantum-Entropic
    def calculate_macro(row_index):
        if row_index < 5:
            return np.nan
        window = data.iloc[row_index-5:row_index+1]
        numerator = ((window['high'] - window['low']) * (window['close'] - window['open'])).sum()
        denominator = window['close'].sum()
        return numerator / denominator * data.iloc[row_index]['volume'] if denominator != 0 else np.nan
    
    macro_quantum_entropic = pd.Series([calculate_macro(i) for i in range(len(data))], index=data.index)
    
    # Quantum-Entropic Coherence
    quantum_entropic_coherence = ((micro_quantum_entropic - meso_quantum_entropic) + 
                                 (meso_quantum_entropic - macro_quantum_entropic))
    
    # Quantum-Entropic Flow Dynamics
    entropic_flow = ((data['close'] - data['close'].shift(1)) * data['volume'] / 
                    (data['high'] - data['low'] + 0.0001))
    
    quantum_flow = (data['amount'] / (data['volume'] + 0.0001) * 
                   (data['close'] - data['open']) / (data['high'] - data['low'] + 0.0001))
    
    quantum_entropic_flow_divergence = entropic_flow - quantum_flow
    
    # Quantum-Entropic Core
    quantum_entropic_core = quantum_entropic_coherence * quantum_entropic_flow_divergence
    
    # Pressure-Volume Entanglement Mechanics
    # Opening-Closing Quantum Analysis
    opening_quantum_pressure = ((data['open'] - data['close'].shift(1)) * data['volume'] / 
                               (np.abs(data['open'] - data['low']) + 0.0001))
    
    closing_quantum_pressure = ((data['close'] - data['open']) * data['volume'] / 
                               (np.abs(data['high'] - data['close']) + 0.0001))
    
    quantum_pressure_entanglement = (opening_quantum_pressure * closing_quantum_pressure * 
                                   data['volume'] / (data['volume'].shift(1) + 0.0001))
    
    quantum_turnover_efficiency = (data['volume'] * data['close']) / (
        data['volume'].shift(1) * data['close'].shift(1) + 0.0001)
    
    pressure_volume_core = quantum_pressure_entanglement * quantum_turnover_efficiency
    
    # Volume-Entropic Pattern
    volume_entropic_ratio = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + 0.0001)
    
    price_volume_entropic = (data['close'] - data['close'].shift(1)) * volume_entropic_ratio
    
    quantum_volume_state = (data['volume'] / (data['volume'].shift(1) + 0.0001) * 
                          data['volume'].shift(1) / (data['volume'].shift(2) + 0.0001) * 
                          data['volume'].shift(2) / (data['volume'].shift(3) + 0.0001))
    
    entropic_quantum_intensity = (price_volume_entropic * quantum_volume_state / 
                                (data['high'] - data['low'] + 0.0001))
    
    # Pressure-Volume Resonance
    pressure_volume_resonance = np.cbrt(pressure_volume_core * entropic_quantum_intensity)
    
    # Quantum-Entropic Regime Detection
    high_quantum_entropic_regime = ((data['high'] - data['low']) > 1.9 * (data['high'].shift(1) - data['low'].shift(1))) & (quantum_pressure_entanglement > 0)
    low_quantum_entropic_regime = ((data['high'] - data['low']) < 0.55 * (data['high'].shift(1) - data['low'].shift(1))) & (quantum_volume_state < 1.1)
    
    quantum_entropic_regime_weight = high_quantum_entropic_regime.astype(int) - low_quantum_entropic_regime.astype(int)
    
    # Quantum Flow-Entropic Dynamics
    flow_quantum_variation = (data['amount'] / (data['amount'].shift(1) + 0.0001) * 
                            data['amount'].shift(1) / (data['amount'].shift(2) + 0.0001))
    
    flow_quantum_efficiency = (data['amount'] / (data['volume'] + 0.0001) * 
                             (data['close'] - data['close'].shift(1)) / 
                             (data['high'] - data['low'] + 0.0001))
    
    quantum_flow_momentum = flow_quantum_variation * flow_quantum_efficiency
    
    flow_quantum_density = (data['amount'] / (data['amount'].shift(1) + data['amount'].shift(2) + 
                           data['amount'].shift(3) + 0.0001) * 3)
    
    quantum_flow_entropic_alignment = (data['close'] - data['open']) * quantum_flow_momentum * flow_quantum_density
    
    # Quantum-Entropic Temporal Synchronization
    # Rolling calculations for max and min
    def rolling_max(series, window):
        return series.rolling(window=window, min_periods=1).max()
    
    def rolling_min(series, window):
        return series.rolling(window=window, min_periods=1).min()
    
    immediate_quantum_entropic = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 0.0001)
    
    short_term_quantum_entropic = (data['close'] - data['close'].shift(3)) / (
        rolling_max(data['high'], 4) - rolling_min(data['low'], 4) + 0.0001)
    
    medium_term_quantum_entropic = (data['close'] - data['close'].shift(5)) / (
        rolling_max(data['high'], 6) - rolling_min(data['low'], 6) + 0.0001)
    
    quantum_entropic_phase_alignment = (np.sign(immediate_quantum_entropic) * 
                                      np.sign(short_term_quantum_entropic) * 
                                      np.sign(medium_term_quantum_entropic))
    
    quantum_entropic_amplitude_coherence = 1 / (1 + np.abs(immediate_quantum_entropic - short_term_quantum_entropic) + 
                                              np.abs(short_term_quantum_entropic - medium_term_quantum_entropic))
    
    quantum_entropic_multi_synchronization = quantum_entropic_phase_alignment * quantum_entropic_amplitude_coherence
    
    # Quantum-Entropic Transition Mechanics
    quantum_entropic_price_transition = (data['close'] > data['high'].shift(1)) & (
        (data['close'] - data['low']) / (data['high'] - data['low'] + 0.0001) > 0.65)
    
    quantum_entropic_volume_transition = (data['volume'] > 1.9 * data['volume'].shift(1)) & (
        data['volume'] / (data['volume'].shift(2) + 0.0001) > 1.55)
    
    quantum_entropic_amount_transition = (data['amount'] > 1.75 * data['amount'].shift(1)) & (
        data['amount'] / (data['amount'].shift(2) + 0.0001) > 1.45)
    
    quantum_entropic_multi_transition_score = (quantum_entropic_price_transition.astype(int) + 
                                             quantum_entropic_volume_transition.astype(int) + 
                                             quantum_entropic_amount_transition.astype(int))
    
    quantum_entropic_transition_intensity = (quantum_entropic_multi_transition_score * 
                                           np.abs(data['close'] - data['open']) / 
                                           (data['high'] - data['low'] + 0.0001))
    
    quantum_entropic_transition_amplifier = 1 + (quantum_entropic_transition_intensity * quantum_entropic_multi_transition_score)
    
    # Quantum-Entropic State Dynamics
    quantum_entropic_state_compression = ((data['close'] - rolling_min(data['low'], 5)) / 
                                        (rolling_max(data['high'], 5) - rolling_min(data['low'], 5) + 0.0001))
    
    quantum_entropic_volume_compression = data['volume'] / (rolling_max(data['volume'], 5) + 0.0001)
    
    quantum_entropic_state_confidence = (quantum_entropic_state_compression * 
                                       quantum_entropic_volume_compression * 
                                       np.sign(micro_quantum_entropic))
    
    # Final Quantum-Entropic Alpha Synthesis
    base_signal = quantum_entropic_core * pressure_volume_resonance
    regime_adjusted = base_signal * quantum_entropic_regime_weight
    flow_enhanced = regime_adjusted * quantum_flow_entropic_alignment
    synchronization_amplified = flow_enhanced * quantum_entropic_multi_synchronization
    transition_modulated = synchronization_amplified * quantum_entropic_transition_amplifier
    state_integrated = transition_modulated * quantum_entropic_state_confidence
    
    # Final alpha with volatility normalization
    final_alpha = state_integrated * (1 / (1 + np.abs((data['high'] - data['low']) / 
                                                    (data['high'].shift(1) - data['low'].shift(1) + 0.0001) - 1)))
    
    return final_alpha
