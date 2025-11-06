import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Entanglement Measurement
    data['Quantum_Price_Oscillation'] = (data['high'] - data['low']) / (data['close'] - data['open'] + 1e-8)
    data['Volume_Quantum_State'] = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3) + 1e-8)
    data['Entanglement_Strength'] = data['Quantum_Price_Oscillation'] * data['Volume_Quantum_State']
    
    # Calculate Quantum Coherence
    quantum_osc_sign = np.sign(data['Quantum_Price_Oscillation'])
    quantum_coherence = pd.Series(index=data.index, dtype=float)
    for i in range(2, len(data)):
        if i >= 2:
            window = quantum_osc_sign.iloc[i-2:i+1]
            coherence = (window.iloc[1] == window.iloc[0]).astype(int) + (window.iloc[2] == window.iloc[1]).astype(int)
            quantum_coherence.iloc[i] = coherence / 3
    data['Quantum_Coherence'] = quantum_coherence
    
    # Temporal Entanglement Patterns
    data['Forward_Backward_Asymmetry'] = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1) + 1e-8)
    data['Temporal_Decay_Factor'] = (data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3) + 1e-8)
    data['Quantum_Temporal_Correlation'] = data['Forward_Backward_Asymmetry'] * data['Temporal_Decay_Factor']
    
    # Calculate Entanglement Persistence
    fb_asym_sign = np.sign(data['Forward_Backward_Asymmetry'])
    entanglement_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(2, len(data)):
        if i >= 2:
            window = fb_asym_sign.iloc[i-2:i+1]
            persistence = (window.iloc[1] == window.iloc[0]).astype(int) + (window.iloc[2] == window.iloc[1]).astype(int)
            entanglement_persistence.iloc[i] = persistence / 3
    data['Entanglement_Persistence'] = entanglement_persistence
    
    # Multi-Dimensional Entanglement
    data['Price_Dimension_Entanglement'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    data['Volume_Dimension_Entanglement'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) * ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8))
    data['Cross_Dimensional_Coupling'] = data['Price_Dimension_Entanglement'] * data['Volume_Dimension_Entanglement']
    
    # Calculate Dimensional Coherence
    price_dim_sign = np.sign(data['Price_Dimension_Entanglement'])
    vol_dim_sign = np.sign(data['Volume_Dimension_Entanglement'])
    dimensional_coherence = pd.Series(index=data.index, dtype=float)
    for i in range(2, len(data)):
        if i >= 2:
            window_price = price_dim_sign.iloc[i-2:i+1]
            window_vol = vol_dim_sign.iloc[i-2:i+1]
            coherence = (window_price.iloc[0] == window_vol.iloc[0]).astype(int) + (window_price.iloc[1] == window_vol.iloc[1]).astype(int) + (window_price.iloc[2] == window_vol.iloc[2]).astype(int)
            dimensional_coherence.iloc[i] = coherence / 3
    data['Dimensional_Coherence'] = dimensional_coherence
    
    # Quantum State Transitions
    data['Bullish_Quantum_Jump'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * (data['volume'] / (data['volume'].shift(1) + 1e-8))
    data['Bearish_Quantum_Collapse'] = ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)) * (data['volume'] / (data['volume'].shift(1) + 1e-8))
    data['Quantum_State_Asymmetry'] = data['Bullish_Quantum_Jump'] - data['Bearish_Quantum_Collapse']
    data['State_Transition_Momentum'] = data['Quantum_State_Asymmetry'] / (data['Quantum_State_Asymmetry'].shift(1) + 1e-8) - 1
    
    # Entanglement Break Detection
    data['Price_Entanglement_Break'] = ((data['high'] / (data['high'].shift(1) + 1e-8) - 1) * (data['low'] / (data['low'].shift(1) + 1e-8) - 1))
    data['Volume_Entanglement_Break'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) - (data['volume'].shift(1) / (data['volume'].shift(2) + 1e-8))
    data['Quantum_Break_Signal'] = data['Price_Entanglement_Break'] * data['Volume_Entanglement_Break']
    
    # Calculate Break Confirmation
    price_break_sign = np.sign(data['Price_Entanglement_Break'])
    vol_break_sign = np.sign(data['Volume_Entanglement_Break'])
    break_confirmation = pd.Series(index=data.index, dtype=float)
    for i in range(2, len(data)):
        if i >= 2:
            window_price = price_break_sign.iloc[i-2:i+1]
            window_vol = vol_break_sign.iloc[i-2:i+1]
            confirmation = (window_price.iloc[0] == window_vol.iloc[0]).astype(int) + (window_price.iloc[1] == window_vol.iloc[1]).astype(int) + (window_price.iloc[2] == window_vol.iloc[2]).astype(int)
            break_confirmation.iloc[i] = confirmation / 3
    data['Break_Confirmation'] = break_confirmation
    
    # Quantum Field Dynamics
    data['Strong_Price_Field'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * ((data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2) + 1e-8))
    data['Weak_Price_Field'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * ((data['high'].shift(2) - data['low'].shift(2)) / (data['high'] - data['low'] + 1e-8))
    
    # Calculate Field Persistence
    close_open_sign = np.sign(data['close'] - data['open'])
    field_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(3, len(data)):
        if i >= 3:
            window = close_open_sign.iloc[i-3:i+1]
            persistence = (window.iloc[1] == window.iloc[0]).astype(int) + (window.iloc[2] == window.iloc[1]).astype(int) + (window.iloc[3] == window.iloc[2]).astype(int)
            field_persistence.iloc[i] = persistence / 4
    data['Field_Persistence'] = field_persistence
    
    # Volume Field Regimes
    volume_ma = (data['volume'].shift(3) + data['volume'].shift(2) + data['volume'].shift(1)) / 3
    data['Volume_Field_Amplification'] = (data['close'] - data['open']) * (data['volume'] / (volume_ma + 1e-8))
    data['Volume_Field_Attenuation'] = (data['close'] - data['open']) * (volume_ma / (data['volume'] + 1e-8))
    data['Field_Volume_Coupling'] = np.sign(data['volume'] - data['volume'].shift(2)) * np.sign(data['close'] - data['open'])
    
    # Quantum Field Interactions
    data['Temporal_Field_Gap'] = ((data['open'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(2) + 1e-8)) * (data['volume'] / (data['volume'].shift(1) + 1e-8))
    data['Spatial_Field_Alignment'] = ((data['high'] - data['close']) - (data['close'] - data['low'])) * (abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8))
    
    prev_ratio = (abs(data['close'].shift(1) - data['open'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * data['volume'].shift(1)
    data['Field_Volume_Resonance'] = (abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * (data['volume'] / (prev_ratio + 1e-8))
    
    # Core Quantum Components
    data['Primary_Quantum_Factor'] = data['Entanglement_Strength'] * data['Quantum_Temporal_Correlation']
    data['Secondary_Quantum_Factor'] = data['Cross_Dimensional_Coupling'] * data['Quantum_State_Asymmetry']
    data['Tertiary_Quantum_Factor'] = data['Quantum_Break_Signal'] * data['Price_Dimension_Entanglement']
    data['Quaternary_Quantum_Factor'] = data['State_Transition_Momentum'] * data['Volume_Dimension_Entanglement']
    
    # Quantum Field Validation
    data['Price_Field_Alignment'] = data['Primary_Quantum_Factor'] * data['Field_Persistence']
    data['Volume_Field_Confirmation'] = data['Secondary_Quantum_Factor'] * data['Field_Volume_Coupling']
    data['Break_Field_Validation'] = data['Tertiary_Quantum_Factor'] * data['Break_Confirmation']
    data['Transition_Field_Confirmation'] = data['Quaternary_Quantum_Factor'] * (data['Volume_Quantum_State'] - 1)
    
    # Quantum Pattern Integration
    data['Multi_Field_Quantum_Fusion'] = data['Strong_Price_Field'] * data['Volume_Field_Amplification']
    data['Field_Resonance_Convergence'] = data['Temporal_Field_Gap'] * data['Spatial_Field_Alignment']
    data['Quantum_Volume_Scaling'] = data['Field_Volume_Resonance'] * data['Volume_Quantum_State']
    
    # Composite Quantum Alpha Construction
    data['Alpha_Component_1'] = data['Price_Field_Alignment'] * data['Quantum_Coherence']
    data['Alpha_Component_2'] = data['Volume_Field_Confirmation'] * data['Volume_Quantum_State']
    data['Alpha_Component_3'] = data['Break_Field_Validation'] * data['Dimensional_Coherence']
    data['Alpha_Component_4'] = data['Transition_Field_Confirmation'] * data['Quantum_Price_Oscillation']
    
    # Final Quantum Entanglement Alpha
    data['Quantum_Entanglement_Alpha'] = (data['Alpha_Component_1'] * data['Alpha_Component_2'] * 
                                        data['Alpha_Component_3'] * data['Alpha_Component_4'] * 
                                        data['Multi_Field_Quantum_Fusion'])
    
    # Return the final alpha factor
    return data['Quantum_Entanglement_Alpha']
