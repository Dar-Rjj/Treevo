import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['range'] = df['high'] - df['low']
    df['close_change'] = df['close'].diff()
    df['volume_ratio'] = df['volume'] / (df['volume'].shift(1) + df['volume'].shift(2) + df['volume'].shift(3)).replace(0, 1)
    df['amount_ratio'] = df['amount'] / df['amount'].shift(1).replace(0, 1)
    
    # Fractal Range Volatility (simplified as normalized range)
    df['fractal_range_volatility'] = df['range'] / df['range'].rolling(window=5).mean().replace(0, 1)
    
    # Volume Fractal Dimension (simplified as volume momentum)
    df['volume_fractal_dimension'] = df['volume'] / df['volume'].rolling(window=3).mean().replace(0, 1)
    df['volume_fractal_ratio'] = df['volume_fractal_dimension'] / df['volume_fractal_dimension'].shift(1).replace(0, 1)
    
    # Short-Fractal Efficiency (simplified as price efficiency)
    df['short_fractal_efficiency'] = (df['close'] - df['open']).abs() / df['range'].replace(0, 1)
    
    # Core fractal quantum components
    df['fractal_wavefunction_collapse'] = ((df['close'] - df['open']) / df['range'].replace(0, 1)) * df['fractal_range_volatility'] * df['volume']
    
    df['quantum_fractal_entanglement'] = (
        (df['close_change'].abs() / df['range'].replace(0, 1) - 
         df['close_change'].shift(1).abs() / df['range'].shift(1).replace(0, 1)).abs() * 
        df['volume_fractal_ratio']
    )
    
    df['fractal_superposition_probability'] = (
        ((df['high'] - np.maximum(df['open'], df['close'])) / df['range'].replace(0, 1)) * 
        ((np.minimum(df['open'], df['close']) - df['low']) / df['range'].replace(0, 1)) * 
        df['short_fractal_efficiency']
    )
    
    # Thermodynamic fractal properties
    df['fractal_market_temperature'] = (df['close_change'].abs() / df['range'].replace(0, 1)) * df['volume_ratio']
    
    df['fractal_pressure_gradient'] = (
        ((df['close'] - df['low']) / df['range'].replace(0, 1) - 
         (df['close'].shift(1) - df['low'].shift(1)) / df['range'].shift(1).replace(0, 1)) * 
        df['volume_fractal_ratio']
    )
    
    df['fractal_entropy_production'] = (
        (df['fractal_range_volatility'] - 1).abs() * 
        np.log((df['fractal_range_volatility'] - 1).abs().replace(0, 1)) * 
        df['amount_ratio']
    )
    
    # Fractal quantum-thermodynamic coupling
    df['fractal_wave_pressure_resonance'] = (
        df['fractal_wavefunction_collapse'] * 
        df['fractal_pressure_gradient'] * 
        df['volume_fractal_ratio']
    )
    
    df['fractal_entanglement_temperature'] = (
        df['quantum_fractal_entanglement'] * 
        df['fractal_market_temperature'] * 
        (df['close_change'].abs() / df['range'].replace(0, 1))
    )
    
    df['fractal_superposition_entropy'] = (
        df['fractal_superposition_probability'] * 
        df['fractal_entropy_production'] * 
        df['short_fractal_efficiency']
    )
    
    # Fractal state transition dynamics
    df['quantum_fractal_transition'] = (
        (df['quantum_fractal_entanglement'] / df['quantum_fractal_entanglement'].shift(1).replace(0, 1)) * 
        (df['fractal_wavefunction_collapse'] - df['fractal_wavefunction_collapse'].shift(1)).abs()
    )
    
    df['thermodynamic_fractal_phase'] = (
        (df['fractal_market_temperature'] / df['fractal_market_temperature'].shift(1).replace(0, 1)) * 
        (df['fractal_pressure_gradient'] - df['fractal_pressure_gradient'].shift(1))
    )
    
    # Fractal momentum consistency
    df['fractal_clean_momentum'] = df['close'].diff(3)
    df['momentum_sign_consistency'] = (
        ((df['fractal_clean_momentum'].shift(1) * df['fractal_clean_momentum'].shift(2) > 0).astype(int) +
         (df['fractal_clean_momentum'].shift(2) * df['fractal_clean_momentum'].shift(3) > 0).astype(int)) / 2
    )
    
    # Fractal quantum weight
    df['fractal_quantum_weight'] = (
        df['quantum_fractal_entanglement'].abs() / 
        (1 + df['quantum_fractal_transition'].abs())
    )
    
    # Calculate regime-specific alphas
    for i in range(3, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Determine fractal states
        quantum_state = 'Coherent' if current['fractal_wavefunction_collapse'] > 0 else 'Decoherent'
        thermal_state = 'Heating' if current['fractal_market_temperature'] > prev['fractal_market_temperature'] else 'Cooling'
        pressure_state = 'High Pressure' if current['fractal_pressure_gradient'] > 0 else 'Low Pressure'
        entropy_state = 'Increasing' if current['fractal_entropy_production'] > prev['fractal_entropy_production'] else 'Decreasing'
        
        # Calculate regime-specific alpha
        if (quantum_state == 'Coherent' and thermal_state == 'Heating' and 
            pressure_state == 'High Pressure' and entropy_state == 'Increasing'):
            alpha = (current['fractal_wave_pressure_resonance'] * 
                    current['fractal_entanglement_temperature'] * 
                    (-current['fractal_superposition_entropy']))
        
        elif (quantum_state == 'Coherent' and thermal_state == 'Cooling' and 
              pressure_state == 'Low Pressure' and entropy_state == 'Decreasing'):
            alpha = (current['quantum_fractal_transition'] * 
                    current['thermodynamic_fractal_phase'] * 
                    current['fractal_superposition_probability'])
        
        elif (quantum_state == 'Decoherent' and thermal_state == 'Heating' and 
              pressure_state == 'Low Pressure' and entropy_state == 'Increasing'):
            alpha = (current['fractal_entropy_production'] * 
                    current['fractal_pressure_gradient'] * 
                    (-current['quantum_fractal_entanglement']))
        
        elif (quantum_state == 'Decoherent' and thermal_state == 'Cooling' and 
              pressure_state == 'High Pressure' and entropy_state == 'Decreasing'):
            alpha = (current['fractal_market_temperature'] * 
                    current['fractal_wavefunction_collapse'] * 
                    current['thermodynamic_fractal_phase'])
        
        else:
            # Default bridge factor combination
            state_phase_coupling = (current['quantum_fractal_transition'] * 
                                  current['fractal_market_temperature'] * 
                                  current['fractal_pressure_gradient'])
            
            entanglement_entropy_bridge = (current['quantum_fractal_entanglement'] * 
                                         current['fractal_entropy_production'] * 
                                         current['fractal_superposition_probability'])
            
            wave_temperature_interface = (current['fractal_wavefunction_collapse'] * 
                                        current['fractal_market_temperature'] * 
                                        (-current['fractal_pressure_gradient']))
            
            alpha = (state_phase_coupling + entanglement_entropy_bridge + wave_temperature_interface) / 3
        
        # Final fractal quantum alpha
        result.iloc[i] = (alpha * 
                         current['fractal_quantum_weight'] * 
                         current['fractal_entanglement_temperature'] * 
                         current['momentum_sign_consistency'])
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
