import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Quantum Microstructure Imbalance Alpha Factor
    Captures quantum-like entanglement between price and volume microstructure
    """
    data = df.copy()
    
    # Basic components with proper shifting to avoid lookahead
    data['volume_ratio_1'] = data['volume'] / data['volume'].shift(1)
    data['volume_ratio_2'] = data['volume'].shift(1) / data['volume'].shift(2)
    data['volume_ratio_3'] = data['volume'].shift(2) / data['volume'].shift(3)
    data['amount_ratio'] = data['amount'] / data['amount'].shift(1)
    data['close_change'] = data['close'] - data['close'].shift(1)
    data['price_range'] = data['high'] - data['low']
    data['prev_price_range'] = data['high'].shift(1) - data['low'].shift(1)
    
    # Quantum Volume Coherence
    data['volume_quantum_state'] = data['volume_ratio_1'] * data['volume_ratio_2'] * data['volume_ratio_3']
    
    # Price-Volume Superposition
    data['price_volume_superposition'] = ((data['close'] - data['open']) * data['volume'] / 
                                         (data['price_range'] * data['amount'])).replace([np.inf, -np.inf], 0)
    
    # Microstructure Entanglement
    data['microstructure_entanglement'] = (np.sign(data['close_change']) * data['volume_ratio_1'] * 
                                          data['amount_ratio']).fillna(0)
    
    # Bid-Ask Pressure Proxy
    data['bid_ask_pressure'] = ((data['close'] - data['low']) * data['volume'] / 
                               ((data['high'] - data['close']) * data['volume'])).replace([np.inf, -np.inf], 0)
    
    # Quantum Flow Imbalance
    def calculate_flow_imbalance(window_data):
        up_volume = window_data[(window_data['close'] > window_data['open'])]['volume'].sum()
        down_volume = window_data[(window_data['close'] < window_data['open'])]['volume'].sum()
        return up_volume / down_volume if down_volume > 0 else 1.0
    
    flow_imbalance_values = []
    for i in range(len(data)):
        if i >= 2:
            window = data.iloc[i-2:i+1].copy()
            flow_imbalance_values.append(calculate_flow_imbalance(window))
        else:
            flow_imbalance_values.append(1.0)
    data['quantum_flow_imbalance'] = flow_imbalance_values
    
    # Microstructure Momentum
    def calculate_microstructure_momentum(window_data, current_volume):
        up_volume = window_data[(window_data['close'] > window_data['open'])]['volume'].sum()
        down_volume = window_data[(window_data['close'] < window_data['open'])]['volume'].sum()
        return (up_volume - down_volume) / current_volume if current_volume > 0 else 0
    
    microstructure_momentum_values = []
    for i in range(len(data)):
        if i >= 2:
            window = data.iloc[i-2:i+1].copy()
            current_vol = data.iloc[i]['volume']
            microstructure_momentum_values.append(calculate_microstructure_momentum(window, current_vol))
        else:
            microstructure_momentum_values.append(0)
    data['microstructure_momentum'] = microstructure_momentum_values
    
    # Quantum Jump Magnitude
    data['quantum_jump_magnitude'] = (np.abs(data['open'] / data['close'].shift(1) - 1) * 
                                     data['volume_ratio_1']).fillna(0)
    
    # Gap Entanglement Persistence
    data['gap_entanglement_persistence'] = (np.sign((data['open'] - data['close'].shift(1)) * 
                                                   (data['close'] - data['open'])) * 
                                           data['volume'] / data['volume'].rolling(window=3, min_periods=1).mean()).fillna(0)
    
    # Multi-Scale Gap Coherence
    data['multi_scale_gap_coherence'] = (((data['close'] - data['open']) / data['price_range']) * 
                                        (np.abs(data['close_change']) / data['prev_price_range']) * 
                                        data['volume_ratio_1']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Quantum State Detection
    data['high_coherence'] = ((data['volume_quantum_state'].between(0.8, 1.2)) & 
                             (data['quantum_flow_imbalance'].between(0.9, 1.1))).astype(int)
    data['low_coherence'] = ((data['volume_quantum_state'] > 1.5) | 
                            (data['volume_quantum_state'] < 0.5)).astype(int)
    
    # Transition state detection using rolling cross detection
    data['volume_cross'] = ((data['volume_quantum_state'] > 1) & 
                           (data['volume_quantum_state'].shift(1) <= 1)) | \
                          ((data['volume_quantum_state'] < 1) & 
                           (data['volume_quantum_state'].shift(1) >= 1))
    data['flow_cross'] = ((data['quantum_flow_imbalance'] > 1) & 
                         (data['quantum_flow_imbalance'].shift(1) <= 1)) | \
                        ((data['quantum_flow_imbalance'] < 1) & 
                         (data['quantum_flow_imbalance'].shift(1) >= 1))
    data['transition_state'] = (data['volume_cross'] | data['flow_cross']).astype(int)
    
    # Entanglement states
    data['strong_entanglement'] = ((data['microstructure_entanglement'] > 0) & 
                                  (data['price_volume_superposition'] > 0)).astype(int)
    data['weak_entanglement'] = ((data['microstructure_entanglement'] < 0) | 
                                (data['price_volume_superposition'] < 0)).astype(int)
    data['decoupling_state'] = (np.sign(data['microstructure_entanglement']) != 
                               np.sign(data['price_volume_superposition'])).astype(int)
    
    # Quantum regime classification
    data['coherent_entanglement'] = (data['high_coherence'] & data['strong_entanglement']).astype(int)
    data['decoupled_chaos'] = (data['low_coherence'] & data['decoupling_state']).astype(int)
    data['quantum_transition'] = (data['transition_state'] | 
                                 ((data['quantum_jump_magnitude'] > 0.02) & 
                                  (data['volume'] > data['volume'].rolling(window=3, min_periods=1).mean()) & 
                                  data['weak_entanglement'])).astype(int)
    
    # Core Quantum Components
    data['quantum_coherence_core'] = (data['volume_quantum_state'] * 
                                     data['microstructure_entanglement'] * 
                                     data['quantum_flow_imbalance'])
    
    data['microstructure_momentum_core'] = (data['microstructure_momentum'] * 
                                           data['price_volume_superposition'] * 
                                           data['bid_ask_pressure'])
    
    data['gap_dynamics_core'] = (data['quantum_jump_magnitude'] * 
                                data['gap_entanglement_persistence'] * 
                                data['multi_scale_gap_coherence'])
    
    # State-based filtering
    data['state_based_filtering'] = (data['volume_quantum_state'] * 
                                    data['microstructure_entanglement'])
    
    # Coherence adjustment
    data['coherence_adjustment'] = (data['quantum_coherence_core'] * 
                                   data['gap_dynamics_core'] * 
                                   data['microstructure_momentum_core'])
    
    # Noise suppression
    data['noise_suppression'] = (np.abs(data['quantum_jump_magnitude']) * 
                                data['volume_ratio_1'] * 
                                data['bid_ask_pressure'])
    
    # Quantum enhancement components
    data['momentum_amplification'] = (data['microstructure_momentum_core'] * 
                                     data['quantum_coherence_core'] * 
                                     data['volume_quantum_state'])
    
    data['entanglement_reinforcement'] = (data['quantum_coherence_core'] * 
                                         data['gap_dynamics_core'] * 
                                         data['microstructure_entanglement'])
    
    data['state_transition_boost'] = (data['gap_dynamics_core'] * 
                                     data['microstructure_momentum_core'] * 
                                     data['quantum_flow_imbalance'])
    
    # Primary quantum selection based on regime
    coherent_factor = (data['coherent_entanglement'] * data['quantum_coherence_core'] * 
                      data['microstructure_momentum_core'])
    
    chaos_factor = (data['decoupled_chaos'] * data['gap_dynamics_core'] * 
                   np.abs(data['price_volume_superposition']))
    
    transition_factor = (data['quantum_transition'] * data['state_transition_boost'] * 
                        data['momentum_amplification'])
    
    # Core quantum integration
    data['core_quantum_integration'] = (data['quantum_coherence_core'] * 
                                       data['microstructure_momentum_core'] * 
                                       data['gap_dynamics_core'])
    
    # Quantum enhancement
    data['quantum_enhancement'] = (data['momentum_amplification'] * 
                                  data['entanglement_reinforcement'] * 
                                  data['state_transition_boost'])
    
    # State-adaptive primary factor
    data['state_adaptive_primary'] = (coherent_factor + chaos_factor + transition_factor + 
                                     data['core_quantum_integration'] + data['quantum_enhancement'])
    
    # Quantum refinement
    data['state_filtering'] = (data['state_adaptive_primary'] * 
                              data['state_based_filtering'] * 
                              data['coherence_adjustment'])
    
    data['noise_optimization'] = (data['state_filtering'] * 
                                 data['noise_suppression'] * 
                                 data['volume_quantum_state'])
    
    # Final quantum consistency
    data['quantum_consistency'] = (data['noise_optimization'] * 
                                  data['microstructure_entanglement'] * 
                                  data['quantum_flow_imbalance'])
    
    # Final alpha output - Quantum Microstructure Imbalance Alpha
    alpha = data['quantum_consistency'].fillna(0)
    
    return alpha
