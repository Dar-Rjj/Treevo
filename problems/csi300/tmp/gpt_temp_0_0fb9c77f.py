import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 9:  # Need at least 9 periods for calculations
            result.iloc[i] = 0
            continue
            
        # Current data
        open_t = df['open'].iloc[i]
        high_t = df['high'].iloc[i]
        low_t = df['low'].iloc[i]
        close_t = df['close'].iloc[i]
        amount_t = df['amount'].iloc[i]
        volume_t = df['volume'].iloc[i]
        
        # Historical data
        close_t1 = df['close'].iloc[i-1]
        close_t3 = df['close'].iloc[i-3]
        close_t5 = df['close'].iloc[i-5]
        open_t1 = df['open'].iloc[i-1]
        high_t1 = df['high'].iloc[i-1]
        low_t1 = df['low'].iloc[i-1]
        amount_t1 = df['amount'].iloc[i-1]
        volume_t1 = df['volume'].iloc[i-1]
        
        # Rolling windows
        high_t4_t = df['high'].iloc[i-4:i+1].max()
        low_t4_t = df['low'].iloc[i-4:i+1].min()
        high_t2_t = df['high'].iloc[i-2:i+1].max()
        low_t2_t = df['low'].iloc[i-2:i+1].min()
        high_t5_t3 = df['high'].iloc[i-5:i-2].max()
        low_t5_t3 = df['low'].iloc[i-5:i-2].min()
        high_t9_t5 = df['high'].iloc[i-9:i-4].max()
        low_t9_t5 = df['low'].iloc[i-9:i-4].min()
        
        # Quantum-Fractal Microstructure
        micro_fractal = abs(close_t - close_t1) / (high_t - low_t + 0.001)
        macro_fractal = abs(close_t - close_t5) / (high_t4_t - low_t4_t + 0.001)
        fractal_ratio = micro_fractal / (macro_fractal + 0.001)
        
        opening_quantum_pressure = (open_t - close_t1) * np.log(volume_t + 1) / (np.log(volume_t1 + 1) + 0.001)
        closing_quantum_pressure = (close_t - open_t) * (high_t - low_t) / (abs(close_t - open_t) + 0.001)
        quantum_fractal_fusion = fractal_ratio * opening_quantum_pressure * closing_quantum_pressure
        
        # Behavioral-Volume Entanglement
        behavioral_efficiency = abs(close_t - open_t) / (high_t - low_t + 0.001)
        directional_asymmetry = (2 * close_t - high_t - low_t) / (high_t - low_t + 0.001)
        
        volume_flow_uncertainty = abs(amount_t - amount_t1) / (amount_t + amount_t1 + 0.001)
        
        vol_diff = volume_t - volume_t1
        volume_behavioral_entanglement = np.sign(close_t - open_t) * np.sign(vol_diff) * vol_diff / (volume_t1 + 0.001)
        
        behavioral_volume_core = behavioral_efficiency * directional_asymmetry * volume_flow_uncertainty * volume_behavioral_entanglement
        
        # Multi-Scale Volatility Dynamics
        quantum_vol_momentum = (close_t - close_t1) * (high_t - low_t) / (high_t1 - low_t1 + 0.001)
        fractal_vol_momentum = (close_t - close_t3) * (high_t2_t - low_t2_t) / (high_t5_t3 - low_t5_t3 + 0.001)
        volatility_phase_diff = quantum_vol_momentum - fractal_vol_momentum
        
        vol_phase_shift = fractal_vol_momentum - (close_t - close_t5) * (high_t4_t - low_t4_t) / (high_t9_t5 - low_t9_t5 + 0.001)
        volatility_hierarchy = volatility_phase_diff * vol_phase_shift
        
        # Flow-Resonance Behavioral Patterns
        amount_sum = df['amount'].iloc[i-3:i].sum() + 0.001
        volume_sum = df['volume'].iloc[i-3:i].sum() + 0.001
        
        flow_amplitude = amount_t / amount_sum * 3
        volume_frequency = volume_t / volume_sum * 3
        flow_resonance_state = flow_amplitude * volume_frequency
        
        opening_flow_resonance = abs(open_t - close_t1) * amount_t / (high_t - low_t + 0.001)
        closing_flow_resonance = abs(close_t - open_t) * amount_t / (high_t - low_t + 0.001)
        behavioral_flow_fusion = flow_resonance_state * opening_flow_resonance * closing_flow_resonance
        
        # Core Quantum-Fractal Synthesis
        quantum_fractal_behavioral_core = quantum_fractal_fusion * behavioral_volume_core
        volatility_flow_integration = volatility_hierarchy * behavioral_flow_fusion
        multi_scale_entanglement = quantum_fractal_behavioral_core * volatility_flow_integration
        base_quantum_fractal_alpha = multi_scale_entanglement * (1 + abs(volume_flow_uncertainty))
        
        # Dynamic Regime Classification
        volatility_state = (high_t - low_t) > (df['high'].iloc[i-5] - df['low'].iloc[i-5])
        flow_state = flow_resonance_state > 0.1
        behavioral_state = behavioral_efficiency > 0.6
        quantum_state = abs(opening_quantum_pressure) > abs(closing_quantum_pressure)
        
        # Adaptive Alpha Selection
        high_volatility_quantum_alpha = base_quantum_fractal_alpha * volatility_hierarchy
        high_flow_behavioral_alpha = base_quantum_fractal_alpha * behavioral_flow_fusion
        quantum_fractal_transition_alpha = (quantum_fractal_fusion + behavioral_volume_core) / 2
        behavioral_volatility_alpha = behavioral_volume_core * volatility_flow_integration
        
        # Regime-Based Alpha Selection
        if volatility_state and flow_state:
            regime_alpha = high_volatility_quantum_alpha
        elif flow_state and behavioral_state:
            regime_alpha = high_flow_behavioral_alpha
        elif quantum_state and volatility_state:
            regime_alpha = behavioral_volatility_alpha
        else:
            regime_alpha = quantum_fractal_transition_alpha
        
        # Quantum-Fractal Signal Enhancement
        quantum_price_breakout = (close_t > high_t1) and (abs(close_t - open_t) / (high_t - low_t + 0.001) > 0.6)
        volume_flow_breakout = (volume_t > 1.8 * volume_t1) and (amount_t > 1.7 * amount_t1)
        volatility_breakout = (high_t - low_t) > 1.8 * (high_t1 - low_t1)
        
        breakout_strength = (quantum_price_breakout + volume_flow_breakout + volatility_breakout) * abs(close_t - close_t1) / (high_t - low_t + 0.001)
        signal_amplifier = 1 + (breakout_strength * flow_resonance_state)
        
        # Final Quantum-Fractal Synthesis
        final_alpha = regime_alpha * signal_amplifier * (1 + abs(directional_asymmetry))
        
        result.iloc[i] = final_alpha
    
    return result
