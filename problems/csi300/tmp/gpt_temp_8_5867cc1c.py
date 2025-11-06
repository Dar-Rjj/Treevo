import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Quantum Gap Microstructure
    data['Quantum_Micro_Gap'] = (np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])) * \
                               np.sign(data['close'] - data['open']) * \
                               (data['close'] - data['close'].shift(2)) * \
                               (data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2))
    
    data['Quantum_Meso_Gap'] = (np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])) * \
                              np.sign(data['close'] - data['open']) * \
                              (data['close'] - data['close'].shift(5)) * \
                              (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))
    
    data['Quantum_Macro_Gap'] = (np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])) * \
                               np.sign(data['close'] - data['open']) * \
                               (data['close'] - data['close'].shift(13)) * \
                               (data['high'] - data['low']) / (data['high'].shift(13) - data['low'].shift(13))
    
    # Quantum Gap Entanglement
    data['Gap_Correlation_Short_Medium'] = np.sign(data['Quantum_Micro_Gap']) * np.sign(data['Quantum_Meso_Gap']) * \
                                          np.minimum(np.abs(data['Quantum_Micro_Gap']), np.abs(data['Quantum_Meso_Gap']))
    
    data['Gap_Correlation_Medium_Long'] = np.sign(data['Quantum_Meso_Gap']) * np.sign(data['Quantum_Macro_Gap']) * \
                                         np.minimum(np.abs(data['Quantum_Meso_Gap']), np.abs(data['Quantum_Macro_Gap']))
    
    data['Quantum_Gap_Coherence'] = (data['Gap_Correlation_Short_Medium'] + data['Gap_Correlation_Medium_Long']) / 2
    
    # Gap Decay Dynamics
    data['Micro_Gap_Decay'] = data['Quantum_Micro_Gap'] - \
                             (np.abs(data['open'].shift(1) - data['close'].shift(2)) / (data['high'].shift(1) - data['low'].shift(1))) * \
                             np.sign(data['close'].shift(1) - data['open'].shift(1)) * \
                             (data['close'].shift(1) - data['close'].shift(3)) * \
                             (data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(3) - data['low'].shift(3))
    
    data['Meso_Gap_Decay'] = data['Quantum_Meso_Gap'] - \
                            (np.abs(data['open'].shift(1) - data['close'].shift(2)) / (data['high'].shift(1) - data['low'].shift(1))) * \
                            np.sign(data['close'].shift(1) - data['open'].shift(1)) * \
                            (data['close'].shift(1) - data['close'].shift(6)) * \
                            (data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(6) - data['low'].shift(6))
    
    data['Gap_Decay_Momentum'] = (data['Micro_Gap_Decay'] + data['Meso_Gap_Decay']) / 2
    
    # Temporal Volume-Gap Flow
    data['Opening_Gap_Flow'] = ((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))) * data['volume']
    data['Intraday_Gap_Flow'] = ((data['close'] - data['open']) / (data['high'] - data['low'])) * data['volume']
    data['Gap_Flow_Divergence'] = data['Opening_Gap_Flow'] - data['Intraday_Gap_Flow']
    
    # Multi-Day Gap Persistence
    data['Gap_Persistence_Short'] = (np.sign(data['close'] - data['open']) * data['volume']).rolling(window=3, min_periods=1).sum()
    data['Gap_Persistence_Medium'] = (np.sign(data['close'] - data['open']) * data['volume']).rolling(window=6, min_periods=1).sum()
    data['Gap_Persistence_Ratio'] = data['Gap_Persistence_Short'] / (data['Gap_Persistence_Medium'] + 0.001)
    
    # Gap Decay Integration
    data['Gap_Flow_Decay'] = data['Gap_Persistence_Short'] - data['Gap_Persistence_Medium']
    data['Gap_Decay_Flow_Correlation'] = np.sign(data['Gap_Flow_Decay']) * np.sign(data['Gap_Decay_Momentum']) * \
                                        np.minimum(np.abs(data['Gap_Flow_Decay']), np.abs(data['Gap_Decay_Momentum']))
    
    data['Temporal_Gap_Score'] = data['Quantum_Gap_Coherence'] * data['Gap_Decay_Flow_Correlation']
    
    # Amount-Volume Gap Quantum States
    data['Gap_Volume_Quantum_Short'] = (data['volume'] / (data['volume'].rolling(window=2, min_periods=1).mean())) * (data['close'] - data['close'].shift(1))
    data['Gap_Volume_Quantum_Medium'] = (data['volume'] / (data['volume'].rolling(window=5, min_periods=1).mean())) * (data['close'] - data['close'].shift(1))
    data['Gap_Volume_Quantum_Ratio'] = data['Gap_Volume_Quantum_Short'] / data['Gap_Volume_Quantum_Medium']
    
    data['Gap_Amount_Quantum'] = data['amount'] / data['volume']
    data['Gap_Amount_Quantum_Change'] = data['Gap_Amount_Quantum'] - data['Gap_Amount_Quantum'].rolling(window=2, min_periods=1).mean()
    data['Gap_Amount_Volume_Entanglement'] = data['Gap_Amount_Quantum_Change'] * data['Gap_Volume_Quantum_Ratio']
    
    # Quantum Gap Flow Imbalance
    positive_mask = data['close'] > data['open']
    negative_mask = data['close'] < data['open']
    
    data['Positive_Gap_Flow'] = data['amount'].rolling(window=4, min_periods=1).apply(lambda x: x[positive_mask.loc[x.index]].sum() if len(positive_mask.loc[x.index]) > 0 else 0)
    data['Negative_Gap_Flow'] = data['amount'].rolling(window=4, min_periods=1).apply(lambda x: x[negative_mask.loc[x.index]].sum() if len(negative_mask.loc[x.index]) > 0 else 0)
    
    data['Quantum_Gap_Flow_Ratio'] = (data['Positive_Gap_Flow'] - data['Negative_Gap_Flow']) / (data['Positive_Gap_Flow'] + data['Negative_Gap_Flow'] + 0.001)
    
    # Fractal Gap Decay Patterns
    data['Gap_Position_Decay_Short'] = ((data['open'] - data['low']) / (data['high'] - data['low'])) * \
                                      np.sign(data['open'] - data['close'].shift(1)) * \
                                      (data['close'] - data['close'].shift(2)) / \
                                      (data['high'].rolling(window=3, min_periods=1).max() - data['low'].rolling(window=3, min_periods=1).min())
    
    data['Gap_Position_Decay_Medium'] = ((data['open'] - data['low']) / (data['high'] - data['low'])) * \
                                       np.sign(data['open'] - data['close'].shift(1)) * \
                                       (data['close'] - data['close'].shift(5)) / \
                                       (data['high'].rolling(window=6, min_periods=1).max() - data['low'].rolling(window=6, min_periods=1).min())
    
    data['Gap_Position_Decay_Ratio'] = data['Gap_Position_Decay_Short'] / data['Gap_Position_Decay_Medium']
    
    data['Volume_Gap_Decay_Short'] = data['volume'] / data['volume'].rolling(window=2, min_periods=1).mean()
    data['Volume_Gap_Decay_Medium'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['Volume_Gap_Decay_Ratio'] = data['Volume_Gap_Decay_Short'] / data['Volume_Gap_Decay_Medium']
    
    data['Position_Volume_Decay_Correlation'] = np.sign(data['Gap_Position_Decay_Ratio'] - 1) * np.sign(data['Volume_Gap_Decay_Ratio'] - 1)
    data['Fractal_Gap_Decay_Score'] = (data['Gap_Position_Decay_Ratio'] + data['Volume_Gap_Decay_Ratio']) / 2
    data['Fractal_Gap_Decay_Momentum'] = data['Fractal_Gap_Decay_Score'] * data['Position_Volume_Decay_Correlation']
    
    # Quantum Gap Entanglement Regimes
    data['Quantum_Gap_Entanglement_Multiplier'] = 1.0
    strong_entanglement = (data['Quantum_Gap_Coherence'] > 0.5) & (data['Gap_Decay_Momentum'] > 0)
    weak_entanglement = (data['Quantum_Gap_Coherence'] < -0.5) & (data['Gap_Decay_Momentum'] < 0)
    data.loc[strong_entanglement, 'Quantum_Gap_Entanglement_Multiplier'] = 1.4
    data.loc[weak_entanglement, 'Quantum_Gap_Entanglement_Multiplier'] = 0.6
    
    # Gap Flow Regime Classification
    data['Gap_Flow_Regime_Multiplier'] = 1.0
    constructive_flow = (data['Gap_Flow_Divergence'] > 0) & (data['Gap_Persistence_Ratio'] > 1)
    destructive_flow = (data['Gap_Flow_Divergence'] < 0) & (data['Gap_Persistence_Ratio'] < 1)
    data.loc[constructive_flow, 'Gap_Flow_Regime_Multiplier'] = 1.3
    data.loc[destructive_flow, 'Gap_Flow_Regime_Multiplier'] = 0.7
    
    # Fractal Gap Decay Regimes
    data['Fractal_Gap_Decay_Multiplier'] = 1.0
    accelerating_fractal = (data['Fractal_Gap_Decay_Momentum'] > 0) & (data['Position_Volume_Decay_Correlation'] > 0)
    decelerating_fractal = (data['Fractal_Gap_Decay_Momentum'] < 0) & (data['Position_Volume_Decay_Correlation'] < 0)
    data.loc[accelerating_fractal, 'Fractal_Gap_Decay_Multiplier'] = 1.2
    data.loc[decelerating_fractal, 'Fractal_Gap_Decay_Multiplier'] = 0.8
    
    # Core Quantum Gap Signals
    data['Primary_Quantum_Gap_Signal'] = data['Temporal_Gap_Score'] * data['Gap_Amount_Volume_Entanglement']
    data['Decay_Enhanced_Gap_Signal'] = data['Primary_Quantum_Gap_Signal'] * data['Fractal_Gap_Decay_Momentum']
    data['Flow_Integrated_Gap_Signal'] = data['Decay_Enhanced_Gap_Signal'] * data['Quantum_Gap_Flow_Ratio']
    
    # Volume & Amount Gap Integration
    data['Volume_Gap_Quantum_Adjustment'] = data['Gap_Volume_Quantum_Ratio'] * 0.18
    data['Amount_Gap_Flow_Adjustment'] = data['Gap_Amount_Quantum_Change'] * 0.15
    data['Gap_Persistence_Adjustment'] = data['Gap_Persistence_Ratio'] * 0.12
    
    # Final Quantum Fractal Gap Alpha
    data['Final_Signal'] = (data['Flow_Integrated_Gap_Signal'] + 
                           data['Volume_Gap_Quantum_Adjustment'] + 
                           data['Amount_Gap_Flow_Adjustment'] + 
                           data['Gap_Persistence_Adjustment']) * \
                          data['Quantum_Gap_Entanglement_Multiplier'] * \
                          data['Gap_Flow_Regime_Multiplier'] * \
                          data['Fractal_Gap_Decay_Multiplier']
    
    # Gap Breakout Validation
    breakout_condition = ((data['open'] > data['high'].shift(1)) & (data['close'] > data['open'])) | \
                        ((data['open'] < data['low'].shift(1)) & (data['close'] < data['open']))
    data.loc[breakout_condition, 'Final_Signal'] = data.loc[breakout_condition, 'Final_Signal'] * 1.25
    
    # Quantum Gap Persistence
    coherence_sign_consistency = (np.sign(data['Quantum_Gap_Coherence'].shift(1)) == np.sign(data['Quantum_Gap_Coherence'])) & \
                                (np.sign(data['Quantum_Gap_Coherence'].shift(2)) == np.sign(data['Quantum_Gap_Coherence']))
    data.loc[coherence_sign_consistency, 'Final_Signal'] = data.loc[coherence_sign_consistency, 'Final_Signal'] * 1.15
    
    return data['Final_Signal']
