import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility-Adjusted Momentum
    data['upside_momentum_vol'] = np.where(data['close'] > data['open'], 
                                         (data['close'] - data['open']) / (data['high'] - data['low']), 0)
    data['downside_momentum_vol'] = np.where(data['close'] < data['open'], 
                                           (data['open'] - data['close']) / (data['high'] - data['low']), 0)
    data['momentum_vol_asymmetry'] = data['upside_momentum_vol'] - data['downside_momentum_vol']
    
    # Regime Shift Detection
    data['vol_regime_change'] = np.sign((data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1))) * np.sign(data['close'] - data['close'].shift(1))
    
    # Momentum Regime Persistence
    close_diff_sign = np.sign(data['close'] - data['close'].shift(1))
    momentum_persistence = []
    for i in range(len(data)):
        if i < 2:
            momentum_persistence.append(0)
        else:
            count = 0
            for j in range(i-2, i+1):
                if j >= 1 and j-1 >= 0:
                    if np.sign(data['close'].iloc[j] - data['close'].iloc[j-1]) == np.sign(data['close'].iloc[j-1] - data['close'].iloc[j-2]):
                        count += 1
            momentum_persistence.append(count / 3)
    data['momentum_regime_persistence'] = momentum_persistence
    
    # Volume-Momentum Relationships
    data['volume_momentum_direction'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['amplitude_volume_divergence'] = (abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])) * (data['volume'] / data['volume'].shift(1))
    
    # Cumulative Volume Divergence
    data['cumulative_volume_divergence'] = data['volume_momentum_direction'].rolling(window=5, min_periods=1).sum()
    
    # Volume Divergence Acceleration
    data['volume_divergence_acceleration'] = data['volume_momentum_direction'] - data['volume_momentum_direction'].shift(1)
    
    # Amplitude Divergence Reversal
    data['amplitude_divergence_reversal'] = np.sign(data['amplitude_volume_divergence']) * np.sign(data['amplitude_volume_divergence'].shift(1))
    
    # Price-Volume Efficiency
    data['bull_market_efficiency'] = np.where(data['close'] > data['close'].shift(1), 
                                            (data['close'] - data['low']) / (data['high'] - data['low']), 0)
    data['bear_market_efficiency'] = np.where(data['close'] < data['close'].shift(1), 
                                            (data['high'] - data['close']) / (data['high'] - data['low']), 0)
    data['efficiency_ratio'] = np.where(data['bear_market_efficiency'] != 0, 
                                      data['bull_market_efficiency'] / data['bear_market_efficiency'], 0)
    
    # Efficiency Transition
    data['efficiency_regime_change'] = np.sign(data['efficiency_ratio'] - data['efficiency_ratio'].shift(1))
    
    # Efficiency Persistence
    efficiency_persistence = []
    for i in range(len(data)):
        if i < 2:
            efficiency_persistence.append(0)
        else:
            count = 0
            for j in range(i-2, i+1):
                if j >= 1 and j-1 >= 0:
                    if np.sign(data['efficiency_ratio'].iloc[j]) == np.sign(data['efficiency_ratio'].iloc[j-1]):
                        count += 1
            efficiency_persistence.append(count / 3)
    data['efficiency_persistence'] = efficiency_persistence
    
    # Asymmetric Momentum Signals
    data['upside_momentum_strength'] = (data['close'] - np.minimum(data['open'], data['close'])) / (data['high'] - data['low'])
    data['downside_momentum_strength'] = (np.maximum(data['open'], data['close']) - data['close']) / (data['high'] - data['low'])
    data['net_directional_momentum'] = data['upside_momentum_strength'] - data['downside_momentum_strength']
    
    data['volatility_adjusted_upside'] = data['upside_momentum_strength'] / (data['high'] - data['low'])
    data['volatility_adjusted_downside'] = data['downside_momentum_strength'] / (data['high'] - data['low'])
    data['asymmetric_volatility_momentum'] = data['volatility_adjusted_upside'] - data['volatility_adjusted_downside']
    
    # Core Regime Shift Factors
    data['divergence_regime_factor'] = data['cumulative_volume_divergence'] * data['momentum_vol_asymmetry']
    data['efficiency_regime_factor'] = data['amplitude_volume_divergence'] * data['efficiency_ratio']
    data['momentum_regime_factor'] = data['net_directional_momentum'] * data['momentum_regime_persistence']
    
    # Persistence-Enhanced Regime Components
    volume_momentum_persistence = []
    for i in range(len(data)):
        if i < 2:
            volume_momentum_persistence.append(0)
        else:
            count = 0
            for j in range(i-2, i+1):
                if j >= 1 and j-1 >= 0:
                    if np.sign(data['volume_momentum_direction'].iloc[j]) == np.sign(data['volume_momentum_direction'].iloc[j-1]):
                        count += 1
            volume_momentum_persistence.append(count / 3)
    data['volume_momentum_persistence'] = volume_momentum_persistence
    
    data['divergence_regime_persistence'] = data['divergence_regime_factor'] * data['volume_momentum_persistence']
    data['efficiency_regime_persistence'] = data['efficiency_regime_factor'] * data['efficiency_persistence']
    data['momentum_regime_transition'] = data['momentum_regime_factor'] * abs(data['momentum_regime_persistence'] - 0.5)
    
    # Multi-Timeframe Regime Shift Dynamics
    data['immediate_regime_momentum'] = ((data['close'] / data['close'].shift(2) - 1) * 
                                       (data['volume'] / data['volume'].shift(2)) * 
                                       data['divergence_regime_factor'])
    
    data['volume_regime_flow'] = ((data['volume'] / data['volume'].shift(2) - 1) * 
                                ((data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2))) * 
                                data['amplitude_volume_divergence'])
    
    data['price_regime_transition'] = ((data['close'] / data['close'].shift(9) - 1) - 
                                     (data['close'].shift(4) / data['close'].shift(9) - 1)) * \
                                    (data['volume'] / data['volume'].shift(9)) * data['efficiency_ratio']
    
    data['volume_regime_transition'] = ((data['volume'] / data['volume'].shift(9) - 1) - 
                                      (data['volume'].shift(4) / data['volume'].shift(9) - 1)) * \
                                     ((data['high'] - data['low']) / (data['high'].shift(9) - data['low'].shift(9))) * \
                                     (data['volume'] / data['volume'].shift(1))
    
    # Cross-Scale Regime Alignment
    data['regime_momentum_alignment'] = data['immediate_regime_momentum'] * data['efficiency_ratio'] * data['volume_regime_flow']
    data['regime_transition_alignment'] = data['price_regime_transition'] * data['momentum_vol_asymmetry'] * data['volume_regime_transition']
    
    # Composite Momentum-Volatility Regime Shift Alpha
    data['core_regime_shift_alpha'] = data['divergence_regime_persistence'] * data['efficiency_regime_persistence'] * data['momentum_regime_transition']
    data['dynamic_regime_weight'] = abs(data['net_directional_momentum']) * (data['volume'] / data['volume'].shift(1)) * ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)))
    data['multi_scale_regime_factor'] = data['regime_momentum_alignment'] * data['regime_transition_alignment'] * (data['volume'] / data['volume'].shift(1))
    
    # Final Alpha Signal
    data['alpha'] = (data['core_regime_shift_alpha'] * data['dynamic_regime_weight'] * 
                   data['efficiency_ratio'] * data['multi_scale_regime_factor'])
    
    return data['alpha']
