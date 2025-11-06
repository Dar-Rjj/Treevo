import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(4, len(df)):
        if i < 4:
            alpha.iloc[i] = 0
            continue
            
        # Current data
        open_t = df['open'].iloc[i]
        high_t = df['high'].iloc[i]
        low_t = df['low'].iloc[i]
        close_t = df['close'].iloc[i]
        volume_t = df['volume'].iloc[i]
        
        # Previous data
        close_t1 = df['close'].iloc[i-1]
        close_t2 = df['close'].iloc[i-2]
        high_t1 = df['high'].iloc[i-1]
        low_t1 = df['low'].iloc[i-1]
        volume_t1 = df['volume'].iloc[i-1]
        volume_t2 = df['volume'].iloc[i-2]
        open_t1 = df['open'].iloc[i-1]
        
        # Entangled Price-Volume States
        # Quantum Price Superposition
        price_state_entanglement = (high_t + low_t) / (2 * close_t1) - 1 if close_t1 != 0 else 0
        
        quantum_price_collapse = abs(close_t - (open_t + high_t + low_t)/3) / (high_t - low_t) if (high_t - low_t) != 0 else 0
        
        superposition_persistence = 0
        for j in range(max(0, i-4), i):
            if j > 0 and (high_t - low_t) != 0:
                superposition_persistence += (df['close'].iloc[j] - df['close'].iloc[j-1]) ** 2 / (high_t - low_t) ** 2
        
        # Volume Quantum States
        volume_diff_sum = 0
        for j in range(max(0, i-4), i):
            if j > 0:
                volume_diff_sum += abs(df['volume'].iloc[j] - df['volume'].iloc[j-1])
        
        volume_state_coherence = volume_t / volume_diff_sum if volume_diff_sum != 0 else 0
        
        quantum_volume_interference = (volume_t - volume_t1) / abs(volume_t1 - volume_t2) if abs(volume_t1 - volume_t2) != 0 else 0
        
        volume_state_collapse = abs(volume_t - volume_t1) / volume_t1 if volume_t1 != 0 else 0
        
        # Price-Volume Entanglement
        entanglement_strength = (close_t - open_t) * np.log(volume_t + 1)
        
        quantum_correlation = np.sign(close_t - close_t1) * np.sign(volume_t - volume_t1)
        
        entanglement_persistence = 0
        for j in range(max(0, i-4), i):
            if j > 0:
                entanglement_persistence += np.sign(df['close'].iloc[j] - df['close'].iloc[j-1]) * np.sign(df['volume'].iloc[j] - df['volume'].iloc[j-1])
        
        # Quantum Field Market Dynamics
        # Price Field Excitations
        quantum_price_fluctuations = (high_t - low_t) / abs(close_t - close_t1) if abs(close_t - close_t1) != 0 else 0
        
        field_excitation_energy = (close_t - open_t) ** 2 / (high_t - low_t) if (high_t - low_t) != 0 else 0
        
        quantum_field_stability = abs(close_t - close_t1) / (high_t1 - low_t1) if (high_t1 - low_t1) != 0 else 0
        
        # Volume Field Dynamics
        volume_sum_prev = 0
        for j in range(max(0, i-4), i):
            volume_sum_prev += df['volume'].iloc[j]
        
        volume_field_intensity = volume_t / volume_sum_prev if volume_sum_prev != 0 else 0
        
        quantum_volume_waves = (volume_t - volume_t1) / volume_t2 if volume_t2 != 0 else 0
        
        field_coherence_length = volume_diff_sum / volume_t if volume_t != 0 else 0
        
        # Coupled Field Interactions
        field_coupling_strength = (high_t - low_t) * np.sqrt(volume_t)
        
        quantum_interference_patterns = (close_t - close_t1) * (volume_t - volume_t1)
        
        field_resonance = (close_t - open_t) / (volume_t ** (1/3)) if volume_t > 0 else 0
        
        # Quantum Decoherence Patterns
        # Price State Decoherence
        quantum_price_decoherence = abs((close_t - close_t1) - (close_t1 - close_t2)) / (high_t - low_t) if (high_t - low_t) != 0 else 0
        
        superposition_breakdown = (high_t - low_t) / abs(close_t - close_t1) if abs(close_t - close_t1) != 0 else 0
        
        decoherence_time_scale = 0
        for j in range(max(0, i-4), i):
            if j > 0 and (high_t - low_t) != 0:
                decoherence_time_scale += abs(df['close'].iloc[j] - df['close'].iloc[j-1]) / (high_t - low_t)
        
        # Volume State Collapse
        volume_quantum_jump = volume_t / volume_t1 - 1 if volume_t1 != 0 else 0
        
        state_measurement_effect = abs(volume_t - volume_t1) / volume_t2 if volume_t2 != 0 else 0
        
        collapse_persistence = volume_diff_sum / volume_t if volume_t != 0 else 0
        
        # Entanglement Decoherence
        current_corr = np.sign(close_t - close_t1) * np.sign(volume_t - volume_t1)
        prev_corr = np.sign(close_t1 - close_t2) * np.sign(volume_t1 - volume_t2)
        quantum_correlation_loss = abs(current_corr - prev_corr)
        
        entanglement_fidelity = entanglement_persistence / 5
        
        decoherence_signal = ((close_t - open_t) / np.sqrt(volume_t) if volume_t > 0 else 0) - ((close_t1 - open_t1) / np.sqrt(volume_t1) if volume_t1 > 0 else 0)
        
        # Quantum Tunneling Effects
        # Price Barrier Tunneling
        quantum_resistance_penetration = (close_t - high_t1) / (high_t1 - low_t1) if (high_t1 - low_t1) != 0 else 0
        
        support_level_tunneling = (close_t - low_t1) / (high_t1 - low_t1) if (high_t1 - low_t1) != 0 else 0
        
        tunneling_probability = abs(close_t - close_t1) / (high_t1 - low_t1) if (high_t1 - low_t1) != 0 else 0
        
        # Volume Flow Tunneling
        volume_quantum_flow = volume_t / volume_t1 if volume_t1 != 0 else 0
        
        flow_barrier_penetration = (volume_t - volume_t1) / abs(volume_t1 - volume_t2) if abs(volume_t1 - volume_t2) != 0 else 0
        
        tunneling_volume = volume_t * (close_t - close_t1) / (high_t - low_t) if (high_t - low_t) != 0 else 0
        
        # Coupled Tunneling Events
        synchronized_tunneling = np.sign(close_t - close_t1) * np.sign(volume_t - volume_t1) * abs(close_t - close_t1) / (high_t - low_t) if (high_t - low_t) != 0 else 0
        
        quantum_tunneling_strength = (close_t - close_t1) * volume_t / (high_t - low_t) if (high_t - low_t) != 0 else 0
        
        tunneling_persistence = 0
        for j in range(max(0, i-4), i):
            if j > 0 and (df['high'].iloc[j] - df['low'].iloc[j]) != 0:
                tunneling_persistence += np.sign(df['close'].iloc[j] - df['close'].iloc[j-1]) * abs(df['close'].iloc[j] - df['close'].iloc[j-1]) / (df['high'].iloc[j] - df['low'].iloc[j])
        
        # Quantum Alpha Synthesis
        # Quantum State Assessment
        entanglement_quality_score = (entanglement_strength + quantum_correlation + entanglement_persistence) / 3
        
        quantum_coherence_measure = (volume_state_coherence + field_coherence_length - quantum_correlation_loss) / 3
        
        state_purity_assessment = (1 - quantum_price_decoherence - volume_state_collapse) / 2
        
        # Quantum Dynamics Integration
        field_interaction_strength = (field_coupling_strength + quantum_interference_patterns + field_resonance) / 3
        
        decoherence_signal_processing = (quantum_correlation_loss + decoherence_signal) / 2
        
        tunneling_effect_integration = (synchronized_tunneling + quantum_tunneling_strength + tunneling_persistence) / 3
        
        # Final Quantum Alpha
        quantum_predictive_signal = (price_state_entanglement + quantum_price_collapse + superposition_persistence) / 3
        
        entanglement_momentum = (entanglement_quality_score + quantum_coherence_measure + state_purity_assessment) / 3
        
        decoherence_adjustment = (decoherence_signal_processing + tunneling_effect_integration) / 2
        
        # Combine all components
        quantum_alpha = (
            quantum_predictive_signal * 0.3 +
            entanglement_momentum * 0.3 +
            field_interaction_strength * 0.2 +
            decoherence_adjustment * 0.2
        )
        
        alpha.iloc[i] = quantum_alpha
    
    # Fill initial values
    alpha = alpha.fillna(0)
    
    return alpha
