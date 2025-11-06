import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Regime Detection
    # Volatility State
    data['volatility_state'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))) * \
                              (data['volume'] / data['volume'].shift(1))
    
    # Trend State
    data['high_5d'] = data['high'].rolling(window=5).max()
    data['low_5d'] = data['low'].rolling(window=5).min()
    data['trend_state'] = (data['close'] - data['close'].shift(5)) / (data['high_5d'] - data['low_5d'])
    
    # Market State
    data['market_state'] = (data['close'] - data['close'].shift(21)) / (data['close'].shift(5) - data['close'].shift(26))
    
    # Adaptive Momentum
    # High Vol Momentum
    data['high_vol_momentum'] = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])) * \
                               (data['volume'] / data['volume'].shift(1))
    
    # Low Vol Momentum
    data['high_3d'] = data['high'].rolling(window=3).max()
    data['low_3d'] = data['low'].rolling(window=3).min()
    data['low_vol_momentum'] = ((data['close'] - data['close'].shift(3)) / (data['high_3d'] - data['low_3d'])) * \
                              (data['volume'] / data['volume'].shift(3))
    
    # Trend Following Momentum
    data['trend_following'] = ((data['close'] - data['close'].shift(8)) / (data['close'].shift(2) - data['close'].shift(10))) * \
                             (data['volume'] / data['volume'].shift(8))
    
    # Price Efficiency
    # Gap Efficiency
    data['gap_efficiency'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Intraday Efficiency
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Range Utilization
    data['range_utilization'] = ((data['close'] - data['low']) / (data['high'] - data['low'])) * \
                               ((data['high'] - data['close']) / (data['high'] - data['low']))
    
    # Volume Dynamics
    # Volume Acceleration
    data['vol_accel_1'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    data['vol_accel_2'] = (data['volume'].shift(1) - data['volume'].shift(2)) / data['volume'].shift(2)
    data['volume_acceleration'] = data['vol_accel_1'] * data['vol_accel_2']
    
    # Volume Persistence
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_persistence'] = 0
    for i in range(1, len(data)):
        if data['volume_ratio'].iloc[i] > 1.1:
            if i > 0 and data['volume_persistence'].iloc[i-1] > 0:
                data['volume_persistence'].iloc[i] = data['volume_persistence'].iloc[i-1] + 1
            else:
                data['volume_persistence'].iloc[i] = 1
        else:
            data['volume_persistence'].iloc[i] = 0
    
    # Volume-Volatility Ratio
    data['volume_vol_ratio'] = (data['volume'] / (data['high'] - data['low'])) * \
                              (data['volume'].shift(1) / (data['high'].shift(1) - data['low'].shift(1)))
    
    # Signal Integration
    # Regime Weighted Momentum
    data['high_vol_weight'] = np.where(data['volatility_state'] > 1.2, data['high_vol_momentum'], 0)
    data['low_vol_weight'] = np.where(data['volatility_state'] < 0.8, data['low_vol_momentum'], 0)
    data['trend_weight'] = np.where(data['trend_state'] > 0.05, data['trend_following'], 0)
    data['regime_weighted_momentum'] = data['high_vol_weight'] + data['low_vol_weight'] + data['trend_weight']
    
    # Efficiency Score
    data['efficiency_score'] = data['gap_efficiency'] * data['intraday_efficiency'] * data['range_utilization']
    
    # Volume Signal
    data['volume_signal'] = data['volume_acceleration'] * data['volume_persistence'] * data['volume_vol_ratio']
    
    # Final Alpha
    data['alpha'] = data['regime_weighted_momentum'] * data['efficiency_score'] * data['volume_signal']
    
    # Handle NaN values
    data['alpha'] = data['alpha'].fillna(0)
    
    return data['alpha']
