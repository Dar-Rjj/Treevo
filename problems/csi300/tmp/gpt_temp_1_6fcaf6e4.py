import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price Momentum Components
    data['close_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['high_momentum'] = data['high'] / data['high'].shift(1) - 1
    data['low_momentum'] = data['low'] / data['low'].shift(1) - 1
    
    # Momentum Acceleration Metrics
    data['close_acceleration'] = (data['close'] / data['close'].shift(1)) / (data['close'].shift(1) / data['close'].shift(2))
    data['high_acceleration'] = (data['high'] / data['high'].shift(1)) / (data['high'].shift(1) / data['high'].shift(2))
    data['low_acceleration'] = (data['low'] / data['low'].shift(1)) / (data['low'].shift(1) / data['low'].shift(2))
    
    # Directional Momentum Divergence
    data['bullish_divergence'] = ((data['high_acceleration'] > data['close_acceleration']) & 
                                 (data['close_acceleration'] > data['low_acceleration'])).astype(int)
    data['bearish_divergence'] = ((data['low_acceleration'] > data['close_acceleration']) & 
                                 (data['close_acceleration'] > data['high_acceleration'])).astype(int)
    data['convergence_signal'] = (abs(data['high_acceleration'] - data['low_acceleration']) < 0.01).astype(int)
    
    # True Range Components
    data['high_low_range'] = data['high'] - data['low']
    data['high_prev_close'] = abs(data['high'] - data['close'].shift(1))
    data['low_prev_close'] = abs(data['low'] - data['close'].shift(1))
    
    # Range Expansion Metrics
    data['range_momentum'] = data['high_low_range'] / data['high_low_range'].shift(1)
    data['range_acceleration'] = data['range_momentum'] / data['range_momentum'].shift(1)
    data['range_volatility'] = data[['high_low_range', 'high_prev_close', 'low_prev_close']].max(axis=1)
    
    # Volatility State
    data['expansion_regime'] = (data['range_acceleration'] > 1.2).astype(int)
    data['compression_regime'] = (data['range_acceleration'] < 0.8).astype(int)
    data['neutral_regime'] = (~data['expansion_regime'] & ~data['compression_regime']).astype(int)
    
    # Volume Momentum Analysis
    data['raw_volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) / (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Volume Persistence
    for i in range(len(data)):
        if i >= 4:
            window = data.iloc[i-4:i+1]
            data.loc[data.index[i], 'volume_persistence'] = (window['volume'] > window['volume'].shift(1)).sum()
        else:
            data.loc[data.index[i], 'volume_persistence'] = np.nan
    
    # Volume Spike Clustering
    for i in range(len(data)):
        if i >= 4:
            window = data.iloc[i-4:i+1]
            data.loc[data.index[i], 'volume_spike_clustering'] = (window['volume'] > 1.8 * window['volume'].shift(1)).sum()
        else:
            data.loc[data.index[i], 'volume_spike_clustering'] = np.nan
    
    data['volume_consistency'] = data['volume'] / data['volume'].shift(1)
    data['volume_state'] = np.sign(data['raw_volume_momentum'] * data['volume_consistency'])
    
    # Volume Regime Classification
    data['high_volume_regime'] = ((data['volume'] > 1.2 * data['volume'].shift(1)) & 
                                 (data['volume_acceleration'] > 1.1)).astype(int)
    data['low_volume_regime'] = ((data['volume'] < 0.8 * data['volume'].shift(1)) & 
                                (data['volume_acceleration'] < 0.9)).astype(int)
    data['normal_volume_regime'] = (~data['high_volume_regime'] & ~data['low_volume_regime']).astype(int)
    
    # Regime Interaction Matrix
    data['high_vol_high_vol'] = (data['expansion_regime'] & data['high_volume_regime']).astype(int)
    data['low_vol_low_vol'] = (data['compression_regime'] & data['low_volume_regime']).astype(int)
    data['mixed_regime'] = (~data['high_vol_high_vol'] & ~data['low_vol_low_vol']).astype(int)
    
    # Regime Score
    data['alignment_score'] = data['high_vol_high_vol'] + data['low_vol_low_vol'] - data['mixed_regime']
    data['strength_score'] = abs(data['range_acceleration']) * abs(data['volume_acceleration'])
    
    # Efficiency Metrics
    data['gap_efficiency'] = abs(data['open'] - data['close'].shift(1)) / data['high_low_range']
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / data['high_low_range']
    data['total_efficiency'] = data['gap_efficiency'] * data['intraday_efficiency']
    
    # Gap Dynamics
    data['opening_gap_magnitude'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_filling_efficiency'] = abs(data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1))
    data['gap_direction_persistence'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    
    # Efficiency-Regime Integration
    data['high_efficiency_signal'] = (data['total_efficiency'] > 0.7).astype(int)
    data['low_efficiency_signal'] = (data['total_efficiency'] < 0.3).astype(int)
    data['efficiency_adjustment'] = 1 / (1 + abs(data['opening_gap_magnitude']))
    
    # Multi-timeframe Confirmation
    data['momentum_1d'] = data['close'] / data['close'].shift(1) - 1
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['short_term_acceleration'] = data['momentum_3d'] / data['momentum_1d']
    
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['medium_term_acceleration'] = data['momentum_10d'] / data['momentum_5d']
    
    # Timeframe Alignment
    data['direction_consistency'] = (np.sign(data['momentum_1d']) == np.sign(data['momentum_5d'])).astype(int)
    data['acceleration_consistency'] = (np.sign(data['short_term_acceleration']) == np.sign(data['medium_term_acceleration'])).astype(int)
    data['convergence_strength'] = abs(data['short_term_acceleration'] - data['medium_term_acceleration'])
    
    # Base Momentum Signal
    data['primary_acceleration'] = data['close_acceleration']
    data['directional_bias'] = data['bullish_divergence'] - data['bearish_divergence']
    data['convergence_adjustment'] = 1 - data['convergence_signal']
    
    # Regime Confirmation
    data['vol_vol_multiplier'] = data['alignment_score'] * data['strength_score']
    data['efficiency_multiplier'] = data['total_efficiency'] * data['efficiency_adjustment']
    data['timeframe_multiplier'] = data['direction_consistency'] * (1 - data['convergence_strength'])
    
    # Risk Management Components
    data['volatility_adjustment'] = 1 / data['range_volatility']
    data['volume_spike_adjustment'] = 1 / (1 + data['volume_spike_clustering'])
    data['gap_risk_adjustment'] = 1 / (1 + data['opening_gap_magnitude'])
    
    # Final Alpha Calculation
    data['raw_signal'] = data['primary_acceleration'] * data['directional_bias'] * data['convergence_adjustment']
    data['regime_confirmation'] = data['vol_vol_multiplier'] * data['efficiency_multiplier'] * data['timeframe_multiplier']
    data['confirmed_signal'] = data['raw_signal'] * data['regime_confirmation']
    data['risk_management'] = data['volatility_adjustment'] * data['volume_spike_adjustment'] * data['gap_risk_adjustment']
    
    data['final_alpha'] = data['confirmed_signal'] * data['risk_management']
    
    return data['final_alpha']
