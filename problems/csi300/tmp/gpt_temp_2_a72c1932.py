import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Volatility-Weighted Convergence Momentum factor
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Framework
    # Multi-timeframe momentum calculation
    data['momentum_1d'] = data['close'] / data['close'].shift(1) - 1
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Momentum persistence analysis
    data['direction_consistent'] = (
        (np.sign(data['momentum_1d']) == np.sign(data['momentum_3d'])) & 
        (np.sign(data['momentum_3d']) == np.sign(data['momentum_5d']))
    )
    data['magnitude_progression'] = data['momentum_3d'] / (data['momentum_1d'] + 0.0001)
    
    # Adaptive Volatility Adjustment
    # Volatility estimation
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    data['smoothed_vol_3d'] = (
        data['intraday_vol'].rolling(window=3, min_periods=1).mean()
    )
    data['vol_trend_5d'] = (data['intraday_vol'] / data['smoothed_vol_3d']) - 1
    
    # Dynamic volatility weighting
    # Volatility regime classification
    conditions = [
        data['intraday_vol'] > 0.04,  # High volatility
        (data['intraday_vol'] > 0.02) & (data['intraday_vol'] <= 0.04),  # Medium volatility
        data['intraday_vol'] <= 0.02  # Low volatility
    ]
    choices = ['high', 'medium', 'low']
    data['vol_regime'] = np.select(conditions, choices, default='medium')
    
    # Regime-adaptive momentum combination
    data['combined_momentum'] = 0.0
    data.loc[data['vol_regime'] == 'high', 'combined_momentum'] = (
        0.4 * data['momentum_1d'] + 0.4 * data['momentum_3d'] + 0.2 * data['momentum_5d']
    )
    data.loc[data['vol_regime'] == 'medium', 'combined_momentum'] = (
        0.3 * data['momentum_1d'] + 0.4 * data['momentum_3d'] + 0.3 * data['momentum_5d']
    )
    data.loc[data['vol_regime'] == 'low', 'combined_momentum'] = (
        0.2 * data['momentum_1d'] + 0.3 * data['momentum_3d'] + 0.5 * data['momentum_5d']
    )
    
    # Volatility-scaled momentum
    data['volatility_scaled_momentum'] = data['combined_momentum'] / (data['intraday_vol'] + 0.0001)
    
    # Convergence Pattern Detection
    # Volume-price alignment
    data['volume_momentum_1d'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_momentum_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    
    data['divergence_1d'] = data['momentum_1d'] - data['volume_momentum_1d']
    data['divergence_3d'] = data['momentum_3d'] - data['volume_momentum_3d']
    data['divergence_5d'] = data['momentum_5d'] - data['volume_momentum_5d']
    
    # Convergence strength assessment
    strong_conv = (
        (data['divergence_1d'].abs() < 0.015) & 
        (data['divergence_3d'].abs() < 0.025) & 
        (data['divergence_5d'].abs() < 0.035)
    )
    moderate_conv = (
        (data['divergence_1d'].abs() < 0.025) | 
        (data['divergence_3d'].abs() < 0.035)
    )
    
    conv_conditions = [strong_conv, moderate_conv & ~strong_conv, ~moderate_conv]
    conv_choices = ['strong', 'moderate', 'weak']
    data['convergence_strength'] = np.select(conv_conditions, conv_choices, default='weak')
    
    # Directional pattern
    pos_alignment = (
        (data['divergence_1d'] > -0.01) & 
        (data['divergence_3d'] > -0.02) & 
        (data['divergence_5d'] > -0.03)
    )
    neg_alignment = (
        (data['divergence_1d'] < 0.01) & 
        (data['divergence_3d'] < 0.02) & 
        (data['divergence_5d'] < 0.03)
    )
    
    dir_conditions = [pos_alignment, neg_alignment, ~pos_alignment & ~neg_alignment]
    dir_choices = ['positive', 'negative', 'mixed']
    data['direction_pattern'] = np.select(dir_conditions, dir_choices, default='mixed')
    
    # Volume acceleration dynamics
    data['acceleration_ratio'] = data['volume_momentum_1d'] / (data['volume_momentum_3d'] + 0.0001)
    
    acc_conditions = [
        data['acceleration_ratio'] > 1.3,
        (data['acceleration_ratio'] >= 0.8) & (data['acceleration_ratio'] <= 1.3),
        data['acceleration_ratio'] < 0.8
    ]
    acc_choices = ['high', 'moderate', 'low']
    data['acceleration_regime'] = np.select(acc_conditions, acc_choices, default='moderate')
    
    # Factor Integration
    # Base momentum signal
    data['factor'] = data['volatility_scaled_momentum']
    
    # Convergence multiplier application
    convergence_multiplier = 1.0
    for idx, row in data.iterrows():
        if row['convergence_strength'] == 'strong':
            if row['direction_pattern'] == 'positive':
                convergence_multiplier = 1.6
            elif row['direction_pattern'] == 'negative':
                convergence_multiplier = 0.4
            else:  # mixed
                convergence_multiplier = 1.1
        elif row['convergence_strength'] == 'moderate':
            if row['direction_pattern'] == 'positive':
                convergence_multiplier = 1.3
            elif row['direction_pattern'] == 'negative':
                convergence_multiplier = 0.7
            else:  # mixed
                convergence_multiplier = 1.0
        else:  # weak
            convergence_multiplier = 0.9
        
        data.loc[idx, 'factor'] *= convergence_multiplier
    
    # Volume acceleration enhancement
    for idx, row in data.iterrows():
        if row['acceleration_regime'] == 'high':
            data.loc[idx, 'factor'] *= 1.4
        elif row['acceleration_regime'] == 'moderate':
            data.loc[idx, 'factor'] *= 1.0
        else:  # low
            data.loc[idx, 'factor'] *= 0.8
    
    # Momentum persistence bonus
    for idx, row in data.iterrows():
        if row['direction_consistent']:
            data.loc[idx, 'factor'] *= 1.2
        else:
            data.loc[idx, 'factor'] *= 0.9
    
    # Return the factor series
    return data['factor']
