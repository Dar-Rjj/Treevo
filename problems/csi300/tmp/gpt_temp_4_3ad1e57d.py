import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Entropic Momentum Cascade alpha factor that combines price entropy, volume dynamics,
    multi-scale momentum, and fractal patterns to predict future stock returns.
    """
    data = df.copy()
    
    # Price Entropy Components
    data['Intraday_Entropy'] = ((data['high'] - data['low']) / (data['close'].shift(1) + 0.001)) * \
                              (np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.001))
    
    data['Overnight_Entropy'] = np.abs(data['open'] - data['close'].shift(1)) / \
                               (data['high'].shift(1) - data['low'].shift(1) + 0.001)
    
    data['Entropic_Dispersion'] = data['Intraday_Entropy'] * data['Overnight_Entropy'] * \
                                 np.sign(data['Intraday_Entropy'] - data['Overnight_Entropy'])
    
    # Volume Entropy Flow
    data['Volume_Entropy'] = (data['volume'] / (data['volume'].shift(1) + 0.001)) * \
                            (np.abs(data['volume'] - data['volume'].shift(1)) / (data['volume'] + 0.001))
    
    data['Volume_Pressure'] = ((data['volume'] - data['volume'].shift(3)) / (data['volume'].shift(3) + 0.001)) * \
                             ((data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 0.001))
    
    data['Entropic_Volume_Momentum'] = data['Volume_Entropy'] * data['Volume_Pressure'] * \
                                      np.sign(data['Volume_Entropy'] - data['Volume_Pressure'])
    
    # Entropic Convergence
    data['Entropic_Pressure_Signal'] = data['Entropic_Dispersion'] * data['Entropic_Volume_Momentum'] * \
                                      (1 - np.abs(data['Entropic_Dispersion'] - data['Entropic_Volume_Momentum']))
    
    # Multi-Scale Momentum Components
    data['Micro_Momentum'] = (data['close'] - data['close'].shift(1)) / \
                            (data['high'].shift(1) - data['low'].shift(1) + 0.001)
    
    data['Meso_Momentum'] = (data['close'] - data['close'].shift(3)) / \
                           (data['high'].rolling(window=4).max().shift(3) - data['low'].rolling(window=4).min().shift(3) + 0.001)
    
    data['Macro_Momentum'] = (data['close'] - data['close'].shift(8)) / \
                            (data['high'].rolling(window=9).max().shift(8) - data['low'].rolling(window=9).min().shift(8) + 0.001)
    
    data['Momentum_Cascade'] = data['Micro_Momentum'] * data['Meso_Momentum'] * data['Macro_Momentum'] * \
                              np.sign(data['Micro_Momentum'] - data['Meso_Momentum'])
    
    # Volume Cascade Dynamics
    data['Volume_Micro'] = data['volume'] / (data['volume'].shift(1) + 0.001) - 1
    data['Volume_Meso'] = data['volume'] / (data['volume'].shift(3) + 0.001) - 1
    data['Volume_Macro'] = data['volume'] / (data['volume'].shift(8) + 0.001) - 1
    
    data['Volume_Cascade'] = data['Volume_Micro'] * data['Volume_Meso'] * data['Volume_Macro'] * \
                            np.sign(data['Volume_Micro'] - data['Volume_Meso'])
    
    # Cascade Synchronization
    data['Cascade_Momentum_Signal'] = data['Momentum_Cascade'] * data['Volume_Cascade'] * \
                                     (1 - np.abs(data['Momentum_Cascade'] - data['Volume_Cascade']))
    
    # Fractal Entropic Patterns
    data['Short_term_Entropic_Efficiency'] = np.abs(data['close'] - data['close'].shift(3)) / \
                                            (np.abs(data['close'] - data['close'].shift(1)) + 
                                             np.abs(data['close'].shift(1) - data['close'].shift(2)) + 
                                             np.abs(data['close'].shift(2) - data['close'].shift(3)) + 0.001)
    
    data['Medium_term_Entropic_Persistence'] = ((data['close'] - data['close'].shift(6)) / (data['close'].shift(6) + 0.001)) * \
                                              ((data['volume'] - data['volume'].shift(6)) / (data['volume'].shift(6) + 0.001))
    
    data['Entropic_Fractal_Coherence'] = data['Short_term_Entropic_Efficiency'] * data['Medium_term_Entropic_Persistence']
    
    # Volume Fractal Entropy
    data['Volume_Fractal_Dispersion'] = (data['volume'] / (data['volume'].shift(2) + 0.001)) - \
                                       (data['volume'] / (data['volume'].shift(4) + 0.001))
    
    data['Volume_Entropic_Persistence'] = (data['volume'] / (data['volume'].shift(7) + 0.001)) * \
                                         (np.abs(data['volume'] - data['volume'].shift(7)) / (data['volume'] + 0.001))
    
    data['Volume_Entropic_Fractal'] = data['Volume_Fractal_Dispersion'] * data['Volume_Entropic_Persistence']
    
    # Entropic Regime Classification
    high_coherence = (data['Entropic_Fractal_Coherence'] > 0.15) & (data['Volume_Entropic_Fractal'] > 0)
    low_coherence = (data['Entropic_Fractal_Coherence'] < 0.08) & (data['Volume_Entropic_Fractal'] < 0)
    transition_entropic = np.abs(data['Entropic_Fractal_Coherence'] - data['Volume_Entropic_Fractal']) > 0.25
    
    # Adaptive Entropic Fusion
    data['Entropic_Base'] = data['Entropic_Pressure_Signal'] * data['Cascade_Momentum_Signal']
    data['Fractal_Entropic_Enhancement'] = data['Entropic_Base'] * data['Entropic_Fractal_Coherence']
    data['Volume_Cascade_Confirmation'] = data['Fractal_Entropic_Enhancement'] * np.sign(data['Volume_Cascade'])
    
    # Entropic Regime Weighting
    regime_weight = np.ones(len(data))
    regime_weight[high_coherence] = 1.4
    regime_weight[low_coherence] = 0.7
    regime_weight[transition_entropic] = data['Entropic_Dispersion'][transition_entropic]
    
    # Entropic Adjustment
    intraday_gt_overnight = data['Intraday_Entropy'] > data['Overnight_Entropy']
    regime_weight[intraday_gt_overnight] *= 1.3
    regime_weight[~intraday_gt_overnight] *= 0.8
    
    data['Entropic_Adjusted_Signal'] = data['Volume_Cascade_Confirmation'] * regime_weight
    
    # Cascade Persistence
    data['Signal_Cascade_Persistence'] = 0
    for i in range(1, len(data)):
        if np.sign(data['Entropic_Adjusted_Signal'].iloc[i]) == np.sign(data['Entropic_Adjusted_Signal'].iloc[i-1]):
            data['Signal_Cascade_Persistence'].iloc[i] = data['Signal_Cascade_Persistence'].iloc[i-1] + 1
        else:
            data['Signal_Cascade_Persistence'].iloc[i] = 1
    
    data['Entropic_Amplitude'] = data['Entropic_Adjusted_Signal'] * np.abs(data['Momentum_Cascade'])
    data['Entropic_Momentum_Alpha'] = data['Entropic_Amplitude'] * data['Signal_Cascade_Persistence']
    
    # Final Alpha Factor
    data['Entropic_Cascade_Divergence'] = data['Entropic_Momentum_Alpha'] * data['Entropic_Pressure_Signal']
    data['Fractal_Confirmed_Entropic_Momentum'] = data['Entropic_Cascade_Divergence'] * data['Entropic_Fractal_Coherence']
    
    return data['Fractal_Confirmed_Entropic_Momentum']
