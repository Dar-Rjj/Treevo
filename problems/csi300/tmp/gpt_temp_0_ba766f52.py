import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns for momentum analysis
    data['returns'] = data['close'].pct_change()
    
    # Dual-Timeframe Momentum Direction
    data['momentum_short'] = data['close'].pct_change(3)
    data['momentum_medium'] = data['close'].pct_change(13)
    
    # Classify momentum states
    conditions = [
        (data['momentum_short'] > 0) & (data['momentum_medium'] > 0),
        (data['momentum_short'] < 0) & (data['momentum_medium'] < 0)
    ]
    choices = ['bullish', 'bearish']
    data['momentum_state'] = np.select(conditions, choices, default='transition')
    
    # Detect Regime Change Points
    data['prev_state'] = data['momentum_state'].shift(1)
    data['state_change'] = data['momentum_state'] != data['prev_state']
    
    # Flag transitions and maintain for 5 days
    data['bullish_transition'] = ((data['state_change']) & 
                                 (data['momentum_state'] == 'bullish')).astype(int)
    data['bearish_transition'] = ((data['state_change']) & 
                                 (data['momentum_state'] == 'bearish')).astype(int)
    
    # Create transition windows
    for i in range(1, 6):
        data[f'bullish_transition_lag_{i}'] = data['bullish_transition'].shift(i)
        data[f'bearish_transition_lag_{i}'] = data['bearish_transition'].shift(i)
    
    data['in_transition_window'] = (
        data['bullish_transition'] | data['bearish_transition'] |
        data['bullish_transition_lag_1'] | data['bearish_transition_lag_1'] |
        data['bullish_transition_lag_2'] | data['bearish_transition_lag_2'] |
        data['bullish_transition_lag_3'] | data['bearish_transition_lag_3'] |
        data['bullish_transition_lag_4'] | data['bearish_transition_lag_4']
    ).astype(int)
    
    # Volume Asymmetry Calculation
    data['up_day_volume'] = np.where(data['returns'] > 0, data['volume'], np.nan)
    data['down_day_volume'] = np.where(data['returns'] < 0, data['volume'], np.nan)
    
    data['avg_up_volume'] = data['up_day_volume'].rolling(window=5, min_periods=1).mean()
    data['avg_down_volume'] = data['down_day_volume'].rolling(window=5, min_periods=1).mean()
    
    # Avoid division by zero
    data['volume_pressure_ratio'] = data['avg_up_volume'] / (data['avg_down_volume'] + 1e-8)
    
    # Volume Momentum Gradient
    data['avg_volume_10d'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_momentum'] = np.log(data['volume'] / (data['avg_volume_10d'] + 1e-8))
    data['volume_acceleration'] = data['volume_momentum'].diff(3)
    
    # Volume Pressure Direction and Strength
    data['volume_pressure_direction'] = np.where(
        (data['volume_pressure_ratio'] > 1) & (data['volume_acceleration'] > 0), 1,
        np.where((data['volume_pressure_ratio'] < 1) & (data['volume_acceleration'] < 0), -1, 0)
    )
    
    data['volume_pressure_strength'] = (
        np.abs(data['volume_acceleration']) * 
        np.log(np.abs(data['volume_pressure_ratio'] - 1) + 1)
    )
    
    data['bidirectional_volume_pressure'] = (
        data['volume_pressure_direction'] * data['volume_pressure_strength']
    )
    
    # Price Range Efficiency
    data['daily_range'] = data['high'] - data['low']
    data['range_utilization'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8)
    data['range_efficiency'] = np.abs(data['range_utilization'] - 0.5)
    
    # Efficiency Persistence
    data['high_efficiency'] = ((data['range_utilization'] > 0.7) | 
                              (data['range_utilization'] < 0.3)).astype(int)
    
    # Calculate efficiency streak
    streak = 0
    efficiency_streak = []
    for val in data['high_efficiency'].values:
        if val == 1:
            streak += 1
        else:
            streak = 0
        efficiency_streak.append(streak)
    
    data['efficiency_streak'] = efficiency_streak
    data['efficiency_persistence'] = np.log(data['efficiency_streak'] + 1)
    
    # Transition-Based Weighting
    data['transition_weight'] = np.where(
        data['in_transition_window'] == 1, 1.5, 0.7
    )
    
    # Combine Volume Pressure with Price Efficiency
    data['volume_efficiency_composite'] = (
        data['bidirectional_volume_pressure'] * data['efficiency_persistence']
    )
    
    # Generate Final Alpha Factor
    data['alpha_factor'] = (
        data['transition_weight'] * 
        data['volume_efficiency_composite'] * 
        np.sign(data['momentum_short'])
    )
    
    # Clean up intermediate columns
    result = data['alpha_factor'].copy()
    
    return result
