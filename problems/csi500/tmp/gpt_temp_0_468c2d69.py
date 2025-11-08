import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume Alignment factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Components
    data['momentum_1d'] = data['close'] - data['close'].shift(1)
    data['momentum_3d'] = data['close'] - data['close'].shift(2)
    data['momentum_5d'] = data['close'] - data['close'].shift(4)
    
    # Momentum Quality
    data['direction_consistency'] = (
        (data['momentum_1d'] > 0) & (data['momentum_3d'] > 0) & (data['momentum_5d'] > 0) |
        (data['momentum_1d'] < 0) & (data['momentum_3d'] < 0) & (data['momentum_5d'] < 0)
    ).astype(int)
    
    data['magnitude_ratio'] = np.where(
        data['momentum_5d'] != 0,
        np.abs(data['momentum_1d']) / np.abs(data['momentum_5d']),
        0
    )
    
    data['acceleration'] = np.where(
        data['momentum_3d'] != 0,
        (data['momentum_1d'] - data['momentum_3d']) / np.abs(data['momentum_3d']),
        0
    )
    
    # Volatility Context
    data['daily_range'] = data['high'] - data['low']
    data['range_3d_avg'] = (data['daily_range'] + data['daily_range'].shift(1) + data['daily_range'].shift(2)) / 3
    data['volatility_regime'] = data['daily_range'] / data['range_3d_avg']
    
    # Volatility-Adjusted Momentum
    data['VAM_1d'] = np.where(
        data['daily_range'] != 0,
        data['momentum_1d'] / data['daily_range'],
        0
    )
    data['VAM_3d'] = np.where(
        data['range_3d_avg'] != 0,
        data['momentum_3d'] / data['range_3d_avg'],
        0
    )
    
    # Volume Dynamics
    data['volume_change'] = data['volume'] - data['volume'].shift(1)
    data['volume_direction'] = np.sign(data['volume_change'])
    
    # Calculate direction streak
    data['volume_direction_streak'] = 0
    for i in range(1, len(data)):
        if data['volume_direction'].iloc[i] == data['volume_direction'].iloc[i-1]:
            data['volume_direction_streak'].iloc[i] = data['volume_direction_streak'].iloc[i-1] + 1
        else:
            data['volume_direction_streak'].iloc[i] = 1
    
    data['persistence_strength'] = np.where(
        data['volume'] != 0,
        data['volume_direction_streak'] * np.abs(data['volume_change']) / data['volume'],
        0
    )
    
    # Volume-Momentum Alignment
    data['alignment_score'] = np.sign(data['momentum_1d']) * np.sign(data['volume_change'])
    data['alignment_consistency'] = 0
    for i in range(1, len(data)):
        if data['alignment_score'].iloc[i] > 0 and data['alignment_score'].iloc[i-1] > 0:
            data['alignment_consistency'].iloc[i] = data['alignment_consistency'].iloc[i-1] + 1
        else:
            data['alignment_consistency'].iloc[i] = 0
    
    data['alignment_confidence'] = data['alignment_consistency'] * np.abs(data['momentum_1d'])
    
    # Multi-Timeframe Integration
    data['weighted_momentum_blend'] = (
        0.5 * data['VAM_1d'] + 
        0.3 * data['VAM_3d'] + 
        0.2 * (data['VAM_1d'] - data['VAM_3d'])
    )
    
    data['base_volume_multiplier'] = 1 + (data['volume_direction_streak'] / 10)
    data['alignment_multiplier'] = 1 + (data['alignment_consistency'] / 8)
    data['combined_volume_effect'] = data['base_volume_multiplier'] * data['alignment_multiplier']
    
    # Regime Adaptation
    data['volatility_scaling'] = np.where(
        data['volatility_regime'] > 1.1, 0.7,
        np.where(data['volatility_regime'] < 0.9, 1.3, 1.0)
    )
    
    data['volume_ratio'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3)
    
    # Calculate momentum persistence
    data['momentum_persistence'] = 0
    for i in range(1, len(data)):
        if np.sign(data['momentum_1d'].iloc[i]) == np.sign(data['momentum_1d'].iloc[i-1]):
            data['momentum_persistence'].iloc[i] = data['momentum_persistence'].iloc[i-1] + 1
        else:
            data['momentum_persistence'].iloc[i] = 0
    
    data['combined_persistence'] = np.minimum(data['momentum_persistence'], data['volume_direction_streak'])
    
    # Final Alpha Construction
    data['momentum_base'] = data['weighted_momentum_blend']
    data['volume_enhanced'] = data['momentum_base'] * data['combined_volume_effect']
    data['regime_adjusted'] = data['volume_enhanced'] * data['volatility_scaling']
    
    data['alpha_value'] = data['regime_adjusted'] * (1 + data['combined_persistence'] / 12)
    
    # Return the alpha factor series
    return data['alpha_value']
