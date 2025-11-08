import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume Convergence Alpha Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Core Momentum Components
    # Price Momentum
    data['ultra_short_momentum'] = data['close'] - data['close'].shift(1)
    data['short_term_momentum'] = data['close'] - data['close'].shift(2)
    data['medium_term_momentum'] = data['close'] - data['close'].shift(4)
    
    # Range Momentum
    data['daily_range'] = data['high'] - data['low']
    data['range_change'] = data['daily_range'] - data['daily_range'].shift(1)
    
    # Range Persistence
    range_expansion = (data['range_change'] > 0).astype(int)
    range_contraction = (data['range_change'] < 0).astype(int)
    data['range_persistence'] = 0
    
    for i in range(1, len(data)):
        if range_expansion.iloc[i] == 1:
            data.iloc[i, data.columns.get_loc('range_persistence')] = data.iloc[i-1, data.columns.get_loc('range_persistence')] + 1 if data.iloc[i-1, data.columns.get_loc('range_persistence')] > 0 else 1
        elif range_contraction.iloc[i] == 1:
            data.iloc[i, data.columns.get_loc('range_persistence')] = data.iloc[i-1, data.columns.get_loc('range_persistence')] - 1 if data.iloc[i-1, data.columns.get_loc('range_persistence')] < 0 else -1
    
    # Volume Dynamics
    # Volume Momentum
    data['volume_change'] = data['volume'] - data['volume'].shift(1)
    data['volume_direction'] = np.sign(data['volume_change'])
    
    # Volume Streak
    data['volume_streak'] = 0
    for i in range(1, len(data)):
        if data['volume_direction'].iloc[i] == data['volume_direction'].iloc[i-1]:
            streak = data['volume_streak'].iloc[i-1] + data['volume_direction'].iloc[i]
            data.iloc[i, data.columns.get_loc('volume_streak')] = streak if abs(streak) <= 10 else np.sign(streak) * 10
        else:
            data.iloc[i, data.columns.get_loc('volume_streak')] = data['volume_direction'].iloc[i]
    
    # Volume-Price Alignment
    data['direction_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume_change'])
    
    # Alignment Persistence
    data['alignment_persistence'] = 0
    for i in range(1, len(data)):
        if data['direction_alignment'].iloc[i] > 0:
            data.iloc[i, data.columns.get_loc('alignment_persistence')] = data.iloc[i-1, data.columns.get_loc('alignment_persistence')] + 1 if data.iloc[i-1, data.columns.get_loc('alignment_persistence')] > 0 else 1
    
    # Alignment Strength
    data['alignment_strength'] = abs(data['close'] - data['close'].shift(1)) * abs(data['volume_change'])
    
    # Volatility Context
    # Range-Based Volatility
    data['short_term_vol'] = data['daily_range'] + data['daily_range'].shift(1) + data['daily_range'].shift(2)
    data['medium_term_vol'] = data['daily_range'].rolling(window=5, min_periods=3).sum()
    data['volatility_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    
    # Volatility Regime
    data['volatility_regime'] = 'normal'
    data.loc[data['volatility_ratio'] > 1.1, 'volatility_regime'] = 'high'
    data.loc[data['volatility_ratio'] < 0.9, 'volatility_regime'] = 'low'
    
    # Factor Construction
    # Base Momentum Signal
    data['multi_timeframe_blend'] = (3 * data['ultra_short_momentum'] + 
                                    2 * data['short_term_momentum'] + 
                                    data['medium_term_momentum']) / 6
    
    data['volume_weighted_momentum'] = data['multi_timeframe_blend'] * (1 + np.log(data['volume'] + 1) / 10)
    
    # Avoid division by zero in range adjustment
    range_adj_denom = np.where(data['daily_range'] != 0, data['daily_range'], 1)
    data['range_adjusted_momentum'] = data['volume_weighted_momentum'] * (1 + data['range_change'] / range_adj_denom)
    
    # Volume Confirmation
    data['volume_persistence_boost'] = data['range_adjusted_momentum'] * (1 + data['volume_streak'] / 8)
    data['alignment_enhancement'] = data['volume_persistence_boost'] * (1 + data['alignment_persistence'] / 6)
    
    # Avoid division by zero in strength multiplier
    strength_denom = np.where(data['alignment_strength'] != 0, data['alignment_strength'] / 1000, 0)
    data['strength_multiplier'] = data['alignment_enhancement'] * (1 + strength_denom)
    
    # Regime Adaptation
    # Volatility Scaling
    data['volatility_scaling'] = 1.0
    data.loc[data['volatility_regime'] == 'high', 'volatility_scaling'] = 0.7
    data.loc[data['volatility_regime'] == 'low', 'volatility_scaling'] = 1.3
    
    # Timeframe Emphasis
    data['timeframe_adjusted'] = data['strength_multiplier']
    
    # High vol: emphasize ultra-short momentum
    high_vol_mask = data['volatility_regime'] == 'high'
    data.loc[high_vol_mask, 'timeframe_adjusted'] = (
        data.loc[high_vol_mask, 'strength_multiplier'] * 0.6 + 
        data.loc[high_vol_mask, 'ultra_short_momentum'] * 0.4
    )
    
    # Low vol: emphasize medium-term momentum
    low_vol_mask = data['volatility_regime'] == 'low'
    data.loc[low_vol_mask, 'timeframe_adjusted'] = (
        data.loc[low_vol_mask, 'strength_multiplier'] * 0.6 + 
        data.loc[low_vol_mask, 'medium_term_momentum'] * 0.4
    )
    
    # Momentum Acceleration
    # Avoid division by zero
    short_term_denom = np.where(data['short_term_momentum'] != 0, abs(data['short_term_momentum']), 1)
    data['acceleration'] = (data['ultra_short_momentum'] - data['short_term_momentum']) / short_term_denom
    data['acceleration_confirmation'] = np.sign(data['acceleration']) * np.sign(data['ultra_short_momentum'])
    data['acceleration_boost'] = 1 + 0.1 * data['acceleration_confirmation']
    
    # Final Composite Signal
    data['regime_adapted_alpha'] = data['timeframe_adjusted'] * data['volatility_scaling'] * data['acceleration_boost']
    
    # Volume-Confirmed Alpha
    volume_confirmation = np.where(data['direction_alignment'] > 0, 1.1, 0.9)
    data['volume_confirmed_alpha'] = data['regime_adapted_alpha'] * volume_confirmation
    
    # Final alpha factor
    alpha_factor = data['volume_confirmed_alpha']
    
    return alpha_factor
