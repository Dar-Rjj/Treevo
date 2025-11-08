import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Scaled Multi-Timeframe Momentum with Volume Persistence alpha factor
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Raw Price Components
    data['intraday_return'] = data['close'] - data['open']
    data['daily_range'] = data['high'] - data['low']
    data['prev_close'] = data['close'].shift(1)
    
    # Multi-Timeframe Momentum Framework
    # Ultra-Short Term (1-2 days)
    data['momentum_1d'] = data['close'] - data['close'].shift(1)
    data['range_1d'] = data['daily_range']
    data['volume_1d'] = data['volume']
    
    # Short-Term (3-5 days)
    data['momentum_3d'] = data['close'] - data['close'].shift(2)
    data['range_3d'] = data['daily_range'] + data['daily_range'].shift(1) + data['daily_range'].shift(2)
    data['volume_3d'] = data['volume'] + data['volume'].shift(1) + data['volume'].shift(2)
    
    # Medium-Term (6-10 days)
    data['momentum_10d'] = data['close'] - data['close'].shift(9)
    data['range_10d'] = data['daily_range'].rolling(window=10, min_periods=1).sum()
    data['volume_10d'] = data['volume'].rolling(window=10, min_periods=1).sum()
    
    # Volatility-Adjusted Components
    # Range-Based Volatility Scaling
    data['short_term_vol'] = data['range_3d'] / 3
    data['medium_term_vol'] = data['range_10d'] / 10
    data['volatility_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    
    # Volatility-Scaled Momentum
    data['vsm_1d'] = data['momentum_1d'] / (data['range_1d'] + 1e-8)
    data['vsm_3d'] = data['momentum_3d'] / (data['range_3d'] + 1e-8)
    data['vsm_10d'] = data['momentum_10d'] / (data['range_10d'] + 1e-8)
    
    # Volatility Regime Classification
    conditions = [
        data['volatility_ratio'] > 1.15,
        (data['volatility_ratio'] >= 0.85) & (data['volatility_ratio'] <= 1.15),
        data['volatility_ratio'] < 0.85
    ]
    choices = ['high', 'normal', 'low']
    data['vol_regime'] = np.select(conditions, choices, default='normal')
    
    # Volume Persistence Analysis
    # Volume Direction Persistence
    data['volume_change_dir'] = np.sign(data['volume'] - data['volume'].shift(1))
    
    # Calculate volume direction streak
    data['volume_streak'] = 1
    for i in range(1, len(data)):
        if data['volume_change_dir'].iloc[i] == data['volume_change_dir'].iloc[i-1]:
            data.loc[data.index[i], 'volume_streak'] = data['volume_streak'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'volume_streak'] = 1
    
    data['volume_persistence_strength'] = data['volume_streak'] * abs(data['volume'] - data['volume'].shift(1))
    
    # Volume-Momentum Alignment
    data['alignment_signal'] = np.sign(data['momentum_1d']) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Calculate alignment streak
    data['alignment_streak'] = 1
    for i in range(1, len(data)):
        if data['alignment_signal'].iloc[i] > 0 and data['alignment_signal'].iloc[i-1] > 0:
            data.loc[data.index[i], 'alignment_streak'] = data['alignment_streak'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'alignment_streak'] = 1
    
    data['alignment_confidence'] = data['alignment_streak'] * abs(data['momentum_1d'])
    
    # Volume Regime Detection
    data['volume_ratio'] = data['volume_3d'] / (data['volume_10d'] + 1e-8)
    
    volume_conditions = [
        data['volume_ratio'] > 1.1,
        (data['volume_ratio'] >= 0.9) & (data['volume_ratio'] <= 1.1),
        data['volume_ratio'] < 0.9
    ]
    volume_choices = ['high', 'normal', 'low']
    data['volume_regime'] = np.select(volume_conditions, volume_choices, default='normal')
    
    # Adaptive Factor Construction
    # Base Momentum Signal
    data['weighted_vsm'] = (4 * data['vsm_1d'] + 3 * data['vsm_3d'] + data['vsm_10d']) / 8
    data['volume_integrated'] = data['weighted_vsm'] * np.log(data['volume'] + 1)
    
    # Persistence Enhancement
    data['volume_persistence_boost'] = data['volume_integrated'] * (1 + data['volume_streak'] / 8)
    data['alignment_boost'] = data['volume_persistence_boost'] * (1 + data['alignment_streak'] / 6)
    
    # Volatility Regime Adaptation
    vol_scaling = {
        'high': 0.6,
        'normal': 1.0,
        'low': 1.4
    }
    data['volatility_scaled'] = data['alignment_boost'] * data['vol_regime'].map(vol_scaling)
    
    # Volume Regime Integration
    volume_scaling = {
        'high': 1.3,
        'normal': 1.0,
        'low': 0.7
    }
    data['volume_regime_scaled'] = data['volatility_scaled'] * data['volume_regime'].map(volume_scaling)
    
    # Recent Momentum Acceleration
    data['acceleration_signal'] = data['vsm_3d'] - data['vsm_10d']
    data['acceleration_direction'] = np.sign(data['acceleration_signal'])
    data['momentum_confirmation'] = data['volume_regime_scaled'] * (1 + 0.15 * data['acceleration_direction'])
    
    # Final Alpha Output
    alpha_factor = data['momentum_confirmation']
    
    return alpha_factor
