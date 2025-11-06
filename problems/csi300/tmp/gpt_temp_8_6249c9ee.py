import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(8, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Orbital Liquidity Dynamics
        if i >= 4:
            perihelion_liquidity = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) * (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-4])
            aphelion_pressure = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / (current_data['high'].iloc[i-3] - current_data['low'].iloc[i-3] + 0.0001)
            orbital_liquidity_field = perihelion_liquidity * aphelion_pressure
        else:
            orbital_liquidity_field = 0
        
        # Gravitational Quantum Momentum
        if i >= 8:
            planetary_quantum_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-2]) * (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-2])
            stellar_quantum_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-8]) * (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-8])
            gravitational_quantum_field = planetary_quantum_momentum * stellar_quantum_momentum
        else:
            gravitational_quantum_field = 0
        
        # Spacetime Entanglement
        if i >= 6:
            curvature_entanglement = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / (current_data['high'].iloc[i-4] - current_data['low'].iloc[i-4] + 0.0001)
            temporal_quantum_warp = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-6]
            spacetime_quantum_field = curvature_entanglement * temporal_quantum_warp
        else:
            spacetime_quantum_field = 0
        
        # Quantum Relativity Patterns
        if i >= 4:
            quantum_time_dilation = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / (current_data['high'].iloc[i-2] - current_data['low'].iloc[i-2] + 0.0001)
            quantum_length_contraction = abs(current_data['open'].iloc[i] - current_data['close'].iloc[i-4]) / (current_data['high'].iloc[i-4] - current_data['low'].iloc[i-4] + 0.0001)
            quantum_relativity_field = quantum_time_dilation * quantum_length_contraction
        else:
            quantum_relativity_field = 0
        
        # Quantum Flow-Gravitational Interaction
        if i >= 2:
            flow_quantum_variation = (current_data['amount'].iloc[i] / current_data['amount'].iloc[i-1]) * (current_data['amount'].iloc[i-1] / current_data['amount'].iloc[i-2])
            flow_quantum_efficiency = (current_data['amount'].iloc[i] / current_data['volume'].iloc[i]) * ((current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / (current_data['high'].iloc[i] - current_data['low'].iloc[i] + 0.0001))
            quantum_flow_gravity_alignment = flow_quantum_variation * flow_quantum_efficiency * gravitational_quantum_field
        else:
            quantum_flow_gravity_alignment = 0
        
        # Quantum Regime Detection
        if i >= 2:
            high_quantum_regime = ((current_data['high'].iloc[i] - current_data['low'].iloc[i]) > 2 * (current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1])) and (aphelion_pressure > 0)
            low_quantum_regime = ((current_data['high'].iloc[i] - current_data['low'].iloc[i]) < 0.5 * (current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1])) and ((current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1]) * (current_data['volume'].iloc[i-1] / current_data['volume'].iloc[i-2]) < 1.0)
            
            if high_quantum_regime:
                high_quantum_multiplier = 1 + ((current_data['high'].iloc[i] - current_data['low'].iloc[i]) / (current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1]) - 1) * aphelion_pressure
            else:
                high_quantum_multiplier = 0
                
            if low_quantum_regime:
                low_quantum_multiplier = 1 - (1 - (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / (current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1])) * ((current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1]) * (current_data['volume'].iloc[i-1] / current_data['volume'].iloc[i-2]))
            else:
                low_quantum_multiplier = 0
                
            normal_quantum_multiplier = 1 if not (high_quantum_regime or low_quantum_regime) else 0
        else:
            high_quantum_multiplier, low_quantum_multiplier, normal_quantum_multiplier = 0, 0, 1
        
        # Quantum Momentum Convergence
        if i >= 5:
            immediate_quantum_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / (current_data['high'].iloc[i] - current_data['low'].iloc[i] + 0.0001)
            
            high_3 = max(current_data['high'].iloc[i-2:i+1])
            low_3 = min(current_data['low'].iloc[i-2:i+1])
            short_term_quantum_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / (high_3 - low_3 + 0.0001)
            
            high_5 = max(current_data['high'].iloc[i-4:i+1])
            low_5 = min(current_data['low'].iloc[i-4:i+1])
            medium_term_quantum_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-5]) / (high_5 - low_5 + 0.0001)
            
            quantum_multi_momentum_convergence = (np.sign(immediate_quantum_momentum) * np.sign(short_term_quantum_momentum) * np.sign(medium_term_quantum_momentum)) / (1 + abs(immediate_quantum_momentum - short_term_quantum_momentum) + abs(short_term_quantum_momentum - medium_term_quantum_momentum))
        else:
            quantum_multi_momentum_convergence = 0
        
        # Quantum Breakout Dynamics
        if i >= 2:
            quantum_price_breakout = (current_data['close'].iloc[i] > current_data['high'].iloc[i-1]) and ((current_data['close'].iloc[i] - current_data['open'].iloc[i]) / (current_data['high'].iloc[i] - current_data['low'].iloc[i] + 0.0001) > 0.6)
            quantum_volume_breakout = (current_data['volume'].iloc[i] > 1.8 * current_data['volume'].iloc[i-1]) and (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-2] > 1.5)
            
            quantum_breakout_amplifier = 1 + ((quantum_price_breakout + quantum_volume_breakout) * abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / (current_data['high'].iloc[i] - current_data['low'].iloc[i] + 0.0001))
        else:
            quantum_breakout_amplifier = 1
        
        # Quantum Collapse Integration
        if i >= 4:
            high_4 = max(current_data['high'].iloc[i-4:i+1])
            low_4 = min(current_data['low'].iloc[i-4:i+1])
            volume_4 = min(current_data['volume'].iloc[i-4:i+1])
            
            quantum_state_collapse = (current_data['close'].iloc[i] - low_4) / (high_4 - low_4 + 0.0001)
            quantum_volume_collapse = current_data['volume'].iloc[i] / volume_4
            quantum_collapse_confidence = quantum_state_collapse * quantum_volume_collapse * np.sign(orbital_liquidity_field)
        else:
            quantum_collapse_confidence = 0
        
        # Final Quantum Gravitational Alpha Synthesis
        if i >= 8:
            core_quantum_gravitational_signal = orbital_liquidity_field * gravitational_quantum_field * spacetime_quantum_field * quantum_relativity_field
            quantum_regime_adaptive_core = core_quantum_gravitational_signal * (high_quantum_multiplier + low_quantum_multiplier + normal_quantum_multiplier)
            quantum_momentum_enhanced = quantum_regime_adaptive_core * quantum_multi_momentum_convergence
            quantum_flow_gravity_momentum = quantum_momentum_enhanced * quantum_flow_gravity_alignment
            quantum_breakout_amplified = quantum_flow_gravity_momentum * quantum_breakout_amplifier
            quantum_gravitational_alpha = quantum_breakout_amplified * quantum_collapse_confidence
            
            result.iloc[i] = quantum_gravitational_alpha
        else:
            result.iloc[i] = 0
    
    return result
