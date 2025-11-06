import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required rolling windows
    df['high_3d'] = df['high'].rolling(window=3, min_periods=1).max()
    df['low_3d'] = df['low'].rolling(window=3, min_periods=1).min()
    df['high_3d_shift3'] = df['high'].shift(3).rolling(window=3, min_periods=1).max()
    df['low_3d_shift3'] = df['low'].shift(3).rolling(window=3, min_periods=1).min()
    df['high_2d'] = df['high'].rolling(window=2, min_periods=1).max()
    df['low_2d'] = df['low'].rolling(window=2, min_periods=1).min()
    df['high_7d'] = df['high'].rolling(window=7, min_periods=1).max()
    df['low_7d'] = df['low'].rolling(window=7, min_periods=1).min()
    df['high_5d'] = df['high'].rolling(window=5, min_periods=1).max()
    df['low_5d'] = df['low'].rolling(window=5, min_periods=1).min()
    
    for i in range(len(df)):
        if i < 8:  # Need at least 8 days of data for all calculations
            result.iloc[i] = 0
            continue
            
        # Quantum-Fractal Entropy Framework
        # Multi-Scale Quantum Pressure
        micro_quantum_pressure = (df['open'].iloc[i] - df['close'].iloc[i-1]) * np.log(df['volume'].iloc[i] + 1) / (np.log(df['volume'].iloc[i-1] + 1) + 1e-8)
        short_term_quantum_pressure = (df['close'].iloc[i] - df['close'].iloc[i-3]) * (df['high_3d'].iloc[i] - df['low_3d'].iloc[i]) / (df['high_3d_shift3'].iloc[i] - df['low_3d_shift3'].iloc[i] + 1e-8)
        quantum_pressure_alignment = micro_quantum_pressure * short_term_quantum_pressure
        
        # Behavioral Entropy Dynamics
        opening_behavioral_entropy = -abs(df['open'].iloc[i] - df['close'].iloc[i-1]) * np.log(abs(df['open'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i-1] - df['low'].iloc[i-1]) + 1e-8)
        closing_behavioral_entropy = -abs(df['close'].iloc[i] - df['open'].iloc[i]) * np.log(abs(df['close'].iloc[i] - df['open'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i]) + 1e-8)
        behavioral_entropy_product = opening_behavioral_entropy * closing_behavioral_entropy
        
        # Quantum-Behavioral Synthesis
        pressure_entropy_coupling = quantum_pressure_alignment * behavioral_entropy_product
        volume_entropy_correlation = df['volume'].iloc[i] * behavioral_entropy_product / (df['high'].iloc[i] - df['low'].iloc[i] + 1e-8)
        quantum_behavioral_alpha = pressure_entropy_coupling * volume_entropy_correlation
        
        # Fractal-Entropy Momentum Hierarchy
        # Multi-Scale Fractal Momentum
        micro_fractal_momentum = abs(df['close'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i] - df['low'].iloc[i] + 1e-8)
        short_term_fractal_momentum = (df['close'].iloc[i] - df['close'].iloc[i-3]) / (df['high_2d'].iloc[i] - df['low_2d'].iloc[i] + 1e-8)
        medium_term_fractal_momentum = (df['close'].iloc[i] - df['close'].iloc[i-8]) / (df['high_7d'].iloc[i] - df['low_7d'].iloc[i] + 1e-8)
        
        # Fractal-Entropy Phase Dynamics
        fractal_phase_difference = micro_fractal_momentum - short_term_fractal_momentum
        entropy_phase_shift = short_term_fractal_momentum - medium_term_fractal_momentum
        fractal_entropy_coherence = fractal_phase_difference * entropy_phase_shift
        
        # Fractal Momentum Integration
        micro_short_fractal_divergence = micro_fractal_momentum / (short_term_fractal_momentum + 0.001)
        short_medium_fractal_convergence = short_term_fractal_momentum / (medium_term_fractal_momentum + 0.001)
        fractal_hierarchy_alpha = fractal_entropy_coherence * micro_short_fractal_divergence * short_medium_fractal_convergence
        
        # Information-Value Cascade Mechanics
        # Multi-Dimensional Information Flow
        price_info_flow = abs(df['close'].iloc[i] / df['close'].iloc[i-1] - 1) - abs(df['close'].iloc[i-1] / df['close'].iloc[i-2] - 1)
        volume_info_flow = df['volume'].iloc[i] / (df['volume'].iloc[i-1] + 1e-8) - df['volume'].iloc[i-1] / (df['volume'].iloc[i-2] + 1e-8)
        value_info_flow = (df['amount'].iloc[i] / (df['volume'].iloc[i] + 1e-8)) / (df['amount'].iloc[i-1] / (df['volume'].iloc[i-1] + 1e-8)) - 1
        entropy_info_flow = (df['high'].iloc[i] - df['low'].iloc[i]) / (abs(df['close'].iloc[i] - df['open'].iloc[i]) + 1e-8) - (df['high'].iloc[i-1] - df['low'].iloc[i-1]) / (abs(df['close'].iloc[i-1] - df['open'].iloc[i-1]) + 1e-8)
        
        # Value Cascade Detection
        positive_value_cascade = (price_info_flow > 0) & (volume_info_flow > 0) & (value_info_flow > 0)
        negative_value_cascade = (price_info_flow < 0) & (volume_info_flow < 0) & (value_info_flow < 0)
        cascade_strength = 0
        if positive_value_cascade:
            cascade_strength = price_info_flow * volume_info_flow * value_info_flow
        elif negative_value_cascade:
            cascade_strength = price_info_flow * volume_info_flow * value_info_flow
        
        # Quantum-Value Integration
        quantum_cascade_pressure = cascade_strength * quantum_behavioral_alpha
        value_flow_efficiency = (df['close'].iloc[i] - df['open'].iloc[i]) * df['volume'].iloc[i] / (df['amount'].iloc[i] + 1e-8) * (df['high'].iloc[i] - df['low'].iloc[i]) / (abs(df['close'].iloc[i] - df['open'].iloc[i]) + 1e-8)
        quantum_value_momentum = quantum_cascade_pressure * value_flow_efficiency
        
        # Fractal Quantum Value Patterns
        # Multi-Scale Quantum Value
        micro_quantum_value = (df['close'].iloc[i] - (df['high'].iloc[i] + df['low'].iloc[i])/2) / (df['high'].iloc[i] - df['low'].iloc[i] + 0.0001) * (df['close'].iloc[i] - df['open'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i] + 1e-8)
        macro_quantum_value = (df['close'].iloc[i] - (df['high'].iloc[i-3] + df['low'].iloc[i-3])/2) / (df['high'].iloc[i-3] - df['low'].iloc[i-3] + 0.0001) * (df['close'].iloc[i] - df['close'].iloc[i-3]) / (df['high_3d'].iloc[i] - df['low_3d'].iloc[i] + 1e-8)
        quantum_value_alignment = micro_quantum_value * macro_quantum_value
        
        # Complexity-Enhanced Quantum Value
        intraday_quantum_complexity = abs(df['close'].iloc[i] - df['open'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i] + 1e-8) * (df['close'].iloc[i] - df['open'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i] + 1e-8)
        gap_quantum_complexity = -abs(df['open'].iloc[i] - df['close'].iloc[i-1]) * np.log(abs(df['open'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i-1] - df['low'].iloc[i-1]) + 1e-8) * (df['open'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i-1] - df['low'].iloc[i-1] + 1e-8)
        vwap_t = df['amount'].iloc[i] / (df['volume'].iloc[i] + 1e-8)
        vwap_t_1 = df['amount'].iloc[i-1] / (df['volume'].iloc[i-1] + 1e-8)
        value_quantum_complexity = vwap_t / (df['close'].iloc[i] + 1e-8) * np.log(vwap_t / (df['close'].iloc[i] + 1e-8) / ((df['high'].iloc[i] - df['low'].iloc[i]) / (abs(df['close'].iloc[i] - df['open'].iloc[i]) + 1e-8) + 1e-8) + 1e-8) * (df['close'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i] - df['low'].iloc[i] + 1e-8)
        
        # Fractal Quantum Integration
        quantum_value_momentum_resonance = quantum_value_momentum * quantum_value_alignment
        price_value_quantum_resonance = quantum_behavioral_alpha * micro_quantum_value
        multi_quantum_value_convergence = quantum_value_momentum_resonance * price_value_quantum_resonance * (intraday_quantum_complexity + gap_quantum_complexity + value_quantum_complexity)
        
        # Quantum-Fractal Breakout Confirmation
        # Multi-Dimensional Breakout Detection
        quantum_price_breakout = (df['close'].iloc[i] > df['high'].iloc[i-1]) & ((df['close'].iloc[i] - df['open'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i] + 0.001) > 0.6)
        value_flow_breakout = (df['volume'].iloc[i] > 1.8 * df['volume'].iloc[i-1]) & (vwap_t > 1.1 * vwap_t_1)
        fractal_breakout = (df['high_2d'].iloc[i] - df['low_2d'].iloc[i]) > 1.5 * (df['high_5d'].iloc[i-3] - df['low_5d'].iloc[i-3])
        
        # Quantum-Fractal Breakout Strength
        multi_breakout_score = quantum_price_breakout + value_flow_breakout + fractal_breakout
        quantum_fractal_intensity = multi_breakout_score * abs(df['close'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i] - df['low'].iloc[i] + 1e-8)
        quantum_fractal_persistence = quantum_fractal_intensity * (df['volume'].iloc[i] / (df['volume'].iloc[i-1] + 1e-8)) * (vwap_t / (vwap_t_1 + 1e-8))
        
        # Signal Amplification
        quantum_fractal_amplifier = 1 + (quantum_fractal_persistence * multi_breakout_score)
        value_consolidation_dampener = 1 - ((df['volume'].iloc[i] < 0.7 * df['volume'].iloc[i-1]) & (vwap_t < 0.9 * vwap_t_1))
        amplified_quantum_fractal_alpha = multi_quantum_value_convergence * quantum_fractal_amplifier * value_consolidation_dampener
        
        # Dynamic Quantum-Fractal Enhancement
        # Value Efficiency Enhancement
        vwap_avg_3d = (vwap_t_1 + (df['amount'].iloc[i-2] / (df['volume'].iloc[i-2] + 1e-8)) + (df['amount'].iloc[i-3] / (df['volume'].iloc[i-3] + 1e-8))) / 3
        value_state_adjustment = amplified_quantum_fractal_alpha * (vwap_t / (vwap_avg_3d + 0.001))
        volume_value_quantum = value_state_adjustment * (df['amount'].iloc[i] / (df['volume'].iloc[i] + 0.001))
        
        # Entropy-Value Adjustment
        entropy_value_state = volume_value_quantum * (df['high'].iloc[i] - df['low'].iloc[i]) / (df['high'].iloc[i-1] - df['low'].iloc[i-1] + 1e-8)
        quantum_value_multiplier = 1 + (entropy_value_state * cascade_strength)
        
        # Final Alpha
        quantum_fractal_microstructure_entropy_alpha = volume_value_quantum * quantum_value_multiplier * cascade_strength
        
        result.iloc[i] = quantum_fractal_microstructure_entropy_alpha
    
    # Fill any remaining NaN values with 0
    result = result.fillna(0)
    
    return result
