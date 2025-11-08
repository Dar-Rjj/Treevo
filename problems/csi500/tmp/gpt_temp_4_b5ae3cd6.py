import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Multi-Timeframe Momentum
    data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_20'] = data['close'] / data['close'].shift(20) - 1
    data['multi_timeframe_momentum'] = data['momentum_5'] - data['momentum_20']
    
    # Volatility Adjustment
    data['volatility_adj'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Volatility-Adjusted Momentum
    data['vol_adj_momentum'] = data['multi_timeframe_momentum'] / (data['volatility_adj'] + 1e-8)
    
    # Efficiency Ratio
    data['efficiency_ratio'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Volume Acceleration
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) - 1
    
    # Price-Volume Correlation
    data['price_volume_corr'] = np.sign(data['multi_timeframe_momentum']) * data['volume_acceleration']
    
    # Volume Flow Persistence
    data['pos_volume_flow'] = data['price_volume_corr'] > 0
    data['volume_flow_persistence'] = data['pos_volume_flow'].astype(int)
    for i in range(1, len(data)):
        if data['pos_volume_flow'].iloc[i]:
            data['volume_flow_persistence'].iloc[i] = data['volume_flow_persistence'].iloc[i-1] + 1
        else:
            data['volume_flow_persistence'].iloc[i] = 0
    
    # Volume Pressure
    data['volume_pressure'] = (data['close'] - data['open']) * data['volume'] / (data['amount'] + 1e-8)
    
    # Opening Gap Efficiency
    data['opening_gap_eff'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Closing Efficiency
    data['closing_eff'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Gap-Closing Divergence
    data['gap_closing_div'] = data['opening_gap_eff'] - data['closing_eff']
    
    # Gap Persistence
    data['gap_persistence'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * \
                             ((data['open'].shift(1) - data['close'].shift(2)) / data['close'].shift(2))
    
    # Volatility Regime
    data['ret'] = data['close'] / data['close'].shift(1) - 1
    data['vol_5d'] = data['ret'].rolling(window=5, min_periods=3).std()
    data['vol_21d'] = data['ret'].rolling(window=21, min_periods=15).std()
    data['volatility_regime'] = np.sign(data['vol_5d'] - data['vol_21d'])
    
    # High-Volatility Component
    data['high_vol_component'] = data['vol_adj_momentum'] * data['volume_flow_persistence'] * (1 + np.abs(data['gap_closing_div']))
    
    # Low-Volatility Component
    data['low_vol_component'] = data['gap_closing_div'] * data['volume_pressure'] * data['efficiency_ratio']
    
    # Regime-Adaptive Core
    data['regime_adaptive_core'] = data['volatility_regime'] * data['high_vol_component'] + \
                                  (1 - np.abs(data['volatility_regime'])) * data['low_vol_component']
    
    # Final Alpha
    data['alpha'] = data['regime_adaptive_core'] * data['gap_persistence'] * data['price_volume_corr']
    
    return data['alpha']
