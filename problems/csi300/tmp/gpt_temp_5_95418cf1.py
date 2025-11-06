import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['prev_close'] = df['close'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    
    # Cross-Asset State Intensity (using rolling mean as peer proxy)
    df['state_intensity'] = (abs(df['open'] - df['prev_close']) / df['prev_close']) * \
                           (abs(df['close'] - df['open']) / (df['high'] - df['low'])) * \
                           (df['volume'] / df['prev_volume'])
    
    df['peer_state_intensity'] = df['state_intensity'].rolling(window=20, min_periods=10).mean()
    df['state_efficiency_divergence'] = df['state_intensity'] - df['peer_state_intensity']
    
    # Cross-Asset Quantum Absorption
    df['state_fill'] = (abs(df['close'] - df['open']) / abs(df['open'] - df['prev_close'])) * \
                      ((df['close'] - df['low']) / (df['high'] - df['low']) - \
                       (df['high'] - df['close']) / (df['high'] - df['low']))
    
    df['opening_imbalance'] = ((df['open'] - df['low']) - (df['high'] - df['open'])) * \
                             (df['volume'] / df['prev_volume'])
    
    # Cross-Asset State Convergence
    df['state_alignment'] = np.sign(df['open'] - df['prev_close']) * \
                           np.sign(df['open'].rolling(window=20).mean() - df['prev_close'].rolling(window=20).mean())
    
    df['tunneling'] = (abs(df['close'] - (df['high'] + df['low'])/2) / (df['high'] - df['low'])) * \
                     df['state_efficiency_divergence']
    
    # Cross-Asset Volume-Quantum Fields
    df['volume_intensity'] = (abs(df['open'] - df['prev_close']) / df['prev_close']) * \
                            (df['volume'] / df['amount'])
    
    df['volume_direction_divergence'] = np.sign(df['volume'] - df['prev_volume']) - \
                                       np.sign(df['volume'].rolling(window=20).mean() - df['prev_volume'].rolling(window=20).mean())
    
    # Cross-Asset Quantum Oscillators
    df['quantum_amplitude'] = ((df['high'] - df['low']) / (df['high'].shift(2) - df['low'].shift(2))) - \
                             ((df['high'].shift(1) - df['low'].shift(1)) / (df['high'].shift(3) - df['low'].shift(3)))
    
    df['peer_quantum_amplitude'] = df['quantum_amplitude'].rolling(window=20, min_periods=10).mean()
    df['quantum_state_divergence'] = df['quantum_amplitude'] - df['peer_quantum_amplitude']
    
    df['quantum_efficiency'] = abs(df['close'] - df['open']) / (0.5 * (df['high'] + df['low']) - df['open'])
    
    # Cross-Asset Volume-Quantum Coupling
    df['volume_quantum_divergence'] = df['volume_direction_divergence'] * df['quantum_state_divergence']
    
    # Cross-Asset Quantum Entropy Framework
    df['state_volatility'] = (abs(df['open'] - df['prev_close']) / df['prev_close']) / \
                            ((df['prev_high'] - df['prev_low']) / df['prev_close'])
    
    df['peer_state_volatility'] = df['state_volatility'].rolling(window=20, min_periods=10).mean()
    df['entropy_divergence'] = df['state_volatility'] / df['peer_state_volatility']
    
    # Quantum Flow Transitions
    df['quantum_flow'] = np.where(df['close'] > df['open'], 
                                 (df['close'] - df['low']) * df['volume'],
                                 (df['high'] - df['close']) * df['volume'])
    
    df['peer_flow_pressure'] = df['quantum_flow'].rolling(window=20, min_periods=10).mean()
    df['relative_flow_dominance'] = df['quantum_flow'] / df['peer_flow_pressure']
    
    # Quantum Persistence
    df['quantum_direction'] = ((df['quantum_flow'] > 0).rolling(window=5, min_periods=3).sum())
    df['peer_quantum_direction'] = df['quantum_direction'].rolling(window=20, min_periods=10).mean()
    df['quantum_persistence_advantage'] = df['quantum_direction'] - df['peer_quantum_direction']
    
    # Multi-Regime Cross-Asset Quantum Factor
    for idx in df.index:
        if idx < df.index[10]:  # Skip early periods for rolling calculations
            continue
            
        row = df.loc[idx]
        
        # Determine entropy regime
        if row['entropy_divergence'] > 1.2 and row['state_efficiency_divergence'] > 0:
            # High Entropy-High Quantum Regime
            core_signal = row['entropy_divergence'] * row['quantum_state_divergence'] * row['quantum_persistence_advantage']
            volume_confirmation = core_signal * row['volume_quantum_divergence']
            regime_factor = volume_confirmation * row['state_efficiency_divergence']
            
        elif row['entropy_divergence'] < 0.8 and row['state_efficiency_divergence'] < 0:
            # Low Entropy-Low Quantum Regime
            core_signal = row['state_fill'] * row['quantum_efficiency']
            flow_integration = core_signal * row['relative_flow_dominance']
            regime_factor = flow_integration * row['quantum_persistence_advantage']
            
        elif row['entropy_divergence'] > 1.2 and row['state_efficiency_divergence'] < 0:
            # High Entropy-Low Quantum Regime
            core_signal = -1 * row['state_efficiency_divergence'] / (1 + row['entropy_divergence'])
            flow_adjustment = core_signal * row['volume_direction_divergence']
            regime_factor = flow_adjustment * row['tunneling']
            
        elif row['entropy_divergence'] < 0.8 and row['state_efficiency_divergence'] > 0:
            # Low Entropy-High Quantum Regime
            core_signal = ((row['close'] / df['close'].shift(3).loc[idx] - 1) * 
                          row['state_efficiency_divergence'] * row['quantum_persistence_advantage'])
            quantum_enhancement = core_signal * row['quantum_state_divergence']
            regime_factor = quantum_enhancement * row['volume_direction_divergence']
            
        else:
            # Convergent Quantum Regime
            state_component = 0.3 * row['state_efficiency_divergence']
            quantum_component = 0.3 * row['quantum_state_divergence'] * row['quantum_persistence_advantage']
            volume_component = 0.2 * row['volume_quantum_divergence']
            flow_component = 0.2 * row['volume_direction_divergence']
            regime_factor = state_component + quantum_component + volume_component + flow_component
        
        # Multi-Timeframe Validation
        immediate_confidence = (np.sign(row['state_efficiency_divergence']) * 
                              np.sign(row['quantum_state_divergence']))
        
        # Short-term consistency (using recent data)
        recent_idx = df.index.get_loc(idx)
        if recent_idx >= 2:
            short_term_data = df.iloc[recent_idx-2:recent_idx+1]
            quantum_stability = short_term_data['quantum_state_divergence'].std()
            volume_consistency = (short_term_data['volume_direction_divergence'] > 0).sum()
            short_term_confidence = quantum_stability * volume_consistency
        else:
            short_term_confidence = 1.0
        
        # Medium-term alignment
        if recent_idx >= 5:
            medium_term_data = df.iloc[recent_idx-5:recent_idx+1]
            directional_quantum = (medium_term_data['quantum_state_divergence'] > 0).sum()
            
            # Simple entropy regime consistency (using state volatility as proxy)
            entropy_stability = (medium_term_data['state_volatility'].diff().abs() < 
                               medium_term_data['state_volatility'].std()).sum()
            medium_term_confidence = directional_quantum * entropy_stability
        else:
            medium_term_confidence = 1.0
        
        # Final factor synthesis
        confidence_weight = (immediate_confidence + short_term_confidence + medium_term_confidence) / 3
        final_factor = regime_factor * confidence_weight * (1 + row['entropy_divergence'])
        
        result.loc[idx] = final_factor
    
    # Fill early NaN values with 0
    result = result.fillna(0)
    
    return result
