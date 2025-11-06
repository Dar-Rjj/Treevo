import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components with error handling
    df = df.copy()
    for i in range(len(df)):
        try:
            # Multi-Scale Fractal Components
            if i >= 1:
                micro_fractal = abs(df['close'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i] - df['low'].iloc[i] + 0.0001)
            else:
                micro_fractal = 0
                
            if i >= 5:
                high_max_5 = max(df['high'].iloc[i-4:i+1])
                low_min_5 = min(df['low'].iloc[i-4:i+1])
                macro_fractal = abs(df['close'].iloc[i] - df['close'].iloc[i-5]) / (high_max_5 - low_min_5 + 0.0001)
            else:
                macro_fractal = 1
                
            fractal_ratio = micro_fractal / (macro_fractal + 0.0001)
            
            # Quantum Pressure Asymmetry
            if i >= 1:
                opening_quantum_pressure = (df['open'].iloc[i] - df['close'].iloc[i-1]) * (df['volume'].iloc[i] - df['volume'].iloc[i-1]) / (df['high'].iloc[i-1] - df['low'].iloc[i-1] + 0.0001)
                closing_quantum_pressure = (df['close'].iloc[i] - df['open'].iloc[i]) * (df['volume'].iloc[i] - df['volume'].iloc[i-1]) / (df['high'].iloc[i] - df['low'].iloc[i] + 0.0001)
                quantum_pressure_asymmetry = opening_quantum_pressure * closing_quantum_pressure
            else:
                quantum_pressure_asymmetry = 0
                
            fractal_quantum_core = fractal_ratio * quantum_pressure_asymmetry
            
            # Quantum Volume Concentration
            if i >= 3:
                quantum_spike_intensity = df['volume'].iloc[i] / (df['volume'].iloc[i-1] + df['volume'].iloc[i-2] + df['volume'].iloc[i-3] + 0.0001) * 3
            else:
                quantum_spike_intensity = 1
                
            if i >= 2:
                quantum_persistence = (df['volume'].iloc[i] / (df['volume'].iloc[i-1] + 0.0001)) * (df['volume'].iloc[i-1] / (df['volume'].iloc[i-2] + 0.0001))
            else:
                quantum_persistence = 1
                
            quantum_concentration = quantum_spike_intensity * quantum_persistence
            
            # Pressure Convergence Analysis
            if i >= 1:
                immediate_pressure = (df['close'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i] - df['low'].iloc[i] + 0.0001)
            else:
                immediate_pressure = 0
                
            if i >= 3:
                high_max_3 = max(df['high'].iloc[i-2:i+1])
                low_min_3 = min(df['low'].iloc[i-2:i+1])
                short_term_pressure = (df['close'].iloc[i] - df['close'].iloc[i-3]) / (high_max_3 - low_min_3 + 0.0001)
            else:
                short_term_pressure = 0
                
            pressure_convergence = np.sign(immediate_pressure) * np.sign(short_term_pressure) / (1 + abs(immediate_pressure - short_term_pressure))
            
            entangled_volume_pressure = quantum_concentration * pressure_convergence
            
            # Quantum Flow Dynamics
            if i >= 2:
                quantum_flow_variation = (df['amount'].iloc[i] / (df['amount'].iloc[i-1] + 0.0001)) * (df['amount'].iloc[i-1] / (df['amount'].iloc[i-2] + 0.0001))
            else:
                quantum_flow_variation = 1
                
            if i >= 1:
                quantum_flow_efficiency = (df['amount'].iloc[i] / (df['volume'].iloc[i] + 0.0001)) * ((df['close'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i] - df['low'].iloc[i] + 0.0001))
            else:
                quantum_flow_efficiency = 0
                
            quantum_flow_momentum = quantum_flow_variation * quantum_flow_efficiency
            
            # Fractal Flow Alignment
            if i >= 1:
                flow_fractal_correlation = np.sign(df['close'].iloc[i] - df['close'].iloc[i-1]) * np.sign(df['amount'].iloc[i] - df['amount'].iloc[i-1]) * fractal_ratio
            else:
                flow_fractal_correlation = 0
                
            fractal_flow_momentum = quantum_flow_momentum * flow_fractal_correlation
            quantum_fractal_flow_composite = fractal_flow_momentum * quantum_concentration
            
            # Quantum Volatility Field
            if i >= 2:
                quantum_amplitude = (df['high'].iloc[i] - df['low'].iloc[i]) * (df['volume'].iloc[i] / (df['volume'].iloc[i-2] + 0.0001))
            else:
                quantum_amplitude = df['high'].iloc[i] - df['low'].iloc[i]
                
            if i >= 1:
                quantum_phase_shift = (df['close'].iloc[i] - df['open'].iloc[i]) / (df['high'].iloc[i-1] - df['low'].iloc[i-1] + 0.0001)
            else:
                quantum_phase_shift = 0
                
            quantum_field = quantum_amplitude * quantum_phase_shift
            
            # Fractal Momentum Components
            if i >= 1:
                short_term_fractal_momentum = (df['close'].iloc[i] - df['close'].iloc[i-1]) * fractal_ratio
            else:
                short_term_fractal_momentum = 0
                
            if i >= 5:
                long_term_fractal_momentum = (df['close'].iloc[i] - df['close'].iloc[i-5]) * macro_fractal
            else:
                long_term_fractal_momentum = 0
                
            fractal_momentum = short_term_fractal_momentum * long_term_fractal_momentum
            entropic_fractal_coupling = quantum_field * fractal_momentum
            
            # Multi-Timeframe Components
            if i >= 1:
                immediate_momentum = (df['close'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i] - df['low'].iloc[i] + 0.0001)
            else:
                immediate_momentum = 0
                
            if i >= 3:
                high_max_3_mom = max(df['high'].iloc[i-2:i+1])
                low_min_3_mom = min(df['low'].iloc[i-2:i+1])
                short_term_momentum = (df['close'].iloc[i] - df['close'].iloc[i-3]) / (high_max_3_mom - low_min_3_mom + 0.0001)
            else:
                short_term_momentum = 0
                
            if i >= 5:
                medium_term_momentum = (df['close'].iloc[i] - df['close'].iloc[i-5]) / (high_max_5 - low_min_5 + 0.0001)
            else:
                medium_term_momentum = 0
                
            # Quantum-Fractal Convergence Detection
            direction_alignment = np.sign(immediate_momentum) * np.sign(short_term_momentum) * np.sign(medium_term_momentum)
            magnitude_coherence = 1 / (1 + abs(immediate_momentum - short_term_momentum) + abs(short_term_momentum - medium_term_momentum))
            multi_momentum_convergence = direction_alignment * magnitude_coherence
            
            # Enhanced Quantum-Fractal Convergence
            strong_convergence = entropic_fractal_coupling * multi_momentum_convergence
            divergence_signal = quantum_fractal_flow_composite * (-1) * (1 - magnitude_coherence)
            adaptive_quantum_fractal_momentum = strong_convergence + divergence_signal
            
            # Microstructure Shock Detection
            if i >= 2:
                price_breakout = int((df['close'].iloc[i] > df['high'].iloc[i-1]) and (abs(df['close'].iloc[i] - df['open'].iloc[i]) / (df['high'].iloc[i] - df['low'].iloc[i] + 0.0001) > 0.6))
                volume_breakout = int((df['volume'].iloc[i] > 1.8 * df['volume'].iloc[i-1]) and (df['volume'].iloc[i] / (df['volume'].iloc[i-2] + 0.0001) > 1.5))
                quantum_volatility_breakout = int(((df['high'].iloc[i] - df['low'].iloc[i]) > 1.8 * (df['high'].iloc[i-1] - df['low'].iloc[i-1])) and (quantum_amplitude > 1.5))
                
                breakout_sum = price_breakout + volume_breakout + quantum_volatility_breakout
                if i >= 1:
                    quantum_breakout_amplifier = 1 + (breakout_sum * abs(df['close'].iloc[i] - df['close'].iloc[i-1]) / (df['high'].iloc[i] - df['low'].iloc[i] + 0.0001) * (df['volume'].iloc[i] / (df['volume'].iloc[i-1] + 0.0001)))
                else:
                    quantum_breakout_amplifier = 1
                    
                quantum_fractal_shock = np.cbrt(quantum_spike_intensity * quantum_breakout_amplifier)
            else:
                quantum_fractal_shock = 1
                
            # Final Alpha Synthesis
            core_quantum_fractal = fractal_quantum_core * entangled_volume_pressure * quantum_fractal_flow_composite
            momentum_enhanced = core_quantum_fractal * adaptive_quantum_fractal_momentum * entropic_fractal_coupling
            shock_integrated = momentum_enhanced * quantum_fractal_shock * quantum_flow_momentum
            final_alpha = shock_integrated * multi_momentum_convergence
            
            result.iloc[i] = final_alpha
            
        except (IndexError, ZeroDivisionError, ValueError):
            result.iloc[i] = 0
    
    # Replace any remaining NaN or inf values
    result = result.replace([np.inf, -np.inf], 0).fillna(0)
    
    return result
