import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Core Price Components
    # Price Momentum
    data['momentum_1d'] = data['close'] - data['close'].shift(1)
    data['momentum_3d'] = data['close'] - data['close'].shift(2)
    data['momentum_5d'] = data['close'] - data['close'].shift(4)
    
    # Price Range
    data['daily_range'] = data['high'] - data['low']
    data['range_3d'] = (data['high'] - data['low']) + \
                       (data['high'].shift(1) - data['low'].shift(1)) + \
                       (data['high'].shift(2) - data['low'].shift(2))
    data['range_5d'] = (data['high'] - data['low']) + \
                       (data['high'].shift(1) - data['low'].shift(1)) + \
                       (data['high'].shift(2) - data['low'].shift(2)) + \
                       (data['high'].shift(3) - data['low'].shift(3)) + \
                       (data['high'].shift(4) - data['low'].shift(4))
    
    # Volume Dynamics
    # Volume Momentum
    data['volume_change'] = data['volume'] - data['volume'].shift(1)
    data['volume_direction'] = np.sign(data['volume_change'])
    data['volume_acceleration'] = (data['volume'] - data['volume'].shift(1)) - \
                                 (data['volume'].shift(1) - data['volume'].shift(2))
    
    # Volume Persistence
    # Direction Streak
    data['volume_direction_shift'] = data['volume_direction'].shift(1)
    data['direction_streak'] = 1
    for i in range(1, len(data)):
        if data['volume_direction'].iloc[i] == data['volume_direction_shift'].iloc[i]:
            data.loc[data.index[i], 'direction_streak'] = data['direction_streak'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'direction_streak'] = 1
    
    data['magnitude_persistence'] = data['volume'] / data['volume'].shift(1)
    
    # Volume Regime
    data['volume_5d_avg'] = (data['volume'] + data['volume'].shift(1) + 
                            data['volume'].shift(2) + data['volume'].shift(3) + 
                            data['volume'].shift(4)) / 5
    data['volume_regime'] = data['volume'] / data['volume_5d_avg']
    
    # Volatility Context
    # Range-Based Volatility
    data['vol_short'] = data['range_3d'] / 3
    data['vol_medium'] = data['range_5d'] / 5
    data['volatility_ratio'] = data['vol_short'] / data['vol_medium']
    
    # Volatility Regime
    data['volatility_regime'] = 'normal'
    data.loc[data['volatility_ratio'] > 1.1, 'volatility_regime'] = 'high'
    data.loc[data['volatility_ratio'] < 0.9, 'volatility_regime'] = 'low'
    
    # Factor Construction
    # Momentum-Volume Alignment
    data['direction_alignment'] = np.sign(data['momentum_1d']) * data['volume_direction']
    data['strength_alignment'] = np.abs(data['momentum_1d']) * np.abs(data['volume_change'])
    data['persistence_alignment'] = data['direction_alignment'] * data['direction_streak']
    
    # Volatility-Scaled Signals
    data['VSM_1d'] = data['momentum_1d'] / data['daily_range']
    data['VSM_3d'] = data['momentum_3d'] / data['range_3d']
    data['VSM_5d'] = data['momentum_5d'] / data['range_5d']
    
    # Adaptive Weighting
    # Volatility-Based Weights
    data['weight_VSM_1d'] = 0.4
    data['weight_VSM_3d'] = 0.4
    data['weight_VSM_5d'] = 0.2
    
    data.loc[data['volatility_regime'] == 'high', 'weight_VSM_1d'] = 0.6
    data.loc[data['volatility_regime'] == 'high', 'weight_VSM_3d'] = 0.3
    data.loc[data['volatility_regime'] == 'high', 'weight_VSM_5d'] = 0.1
    
    data.loc[data['volatility_regime'] == 'low', 'weight_VSM_1d'] = 0.2
    data.loc[data['volatility_regime'] == 'low', 'weight_VSM_3d'] = 0.2
    data.loc[data['volatility_regime'] == 'low', 'weight_VSM_5d'] = 0.6
    
    # Volume-Regime Multipliers
    data['volume_multiplier'] = 1.0
    data.loc[data['volume_regime'] > 1.1, 'volume_multiplier'] = 1.2
    data.loc[data['volume_regime'] < 0.9, 'volume_multiplier'] = 0.8
    
    # Persistence Enhancement
    data['persistence_multiplier'] = 1.0
    data.loc[data['direction_streak'] >= 3, 'persistence_multiplier'] = 1 + data['direction_streak'] / 10
    data.loc[data['direction_streak'] == 1, 'persistence_multiplier'] = 0.9
    
    # Final Alpha Output
    # Base Factor
    data['base_factor'] = (data['weight_VSM_1d'] * data['VSM_1d'] + 
                          data['weight_VSM_3d'] * data['VSM_3d'] + 
                          data['weight_VSM_5d'] * data['VSM_5d'])
    
    # Volume-Adjusted
    data['volume_adjusted'] = data['base_factor'] * data['volume_multiplier']
    
    # Persistence-Enhanced (Final Factor)
    data['alpha_factor'] = data['volume_adjusted'] * data['persistence_multiplier']
    
    # Return the alpha factor series
    return data['alpha_factor']
