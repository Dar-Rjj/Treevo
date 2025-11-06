import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    High-Low Breakout Asymmetry with Volume-Price Confirmation alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily ranges and previous highs/lows
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['daily_range'] = data['high'] - data['low']
    data['avg_range_5'] = data['daily_range'].rolling(window=5).mean()
    
    # Breakout Detection
    data['up_breakout'] = (data['high'] > data['prev_high']).astype(int)
    data['down_breakout'] = (data['low'] < data['prev_low']).astype(int)
    
    # Breakout Strength
    data['up_breakout_strength'] = np.where(
        data['up_breakout'] == 1,
        (data['high'] - data['prev_high']) / data['prev_high'],
        0
    )
    data['down_breakout_strength'] = np.where(
        data['down_breakout'] == 1,
        (data['prev_low'] - data['low']) / data['prev_low'],
        0
    )
    
    # 5-day breakout imbalance
    data['up_breakout_count_5'] = data['up_breakout'].rolling(window=5).sum()
    data['down_breakout_count_5'] = data['down_breakout'].rolling(window=5).sum()
    data['net_breakout_ratio'] = (
        (data['up_breakout_count_5'] - data['down_breakout_count_5']) / 
        (data['up_breakout_count_5'] + data['down_breakout_count_5'] + 1e-8)
    )
    
    # Breakout momentum persistence
    # Consecutive up breakout streak
    data['up_streak'] = 0
    streak = 0
    for i in range(1, len(data)):
        if data['up_breakout'].iloc[i] == 1:
            streak += 1
        else:
            streak = 0
        data.loc[data.index[i], 'up_streak'] = streak
    
    # Consecutive down breakout streak
    data['down_streak'] = 0
    streak = 0
    for i in range(1, len(data)):
        if data['down_breakout'].iloc[i] == 1:
            streak += 1
        else:
            streak = 0
        data.loc[data.index[i], 'down_streak'] = streak
    
    # Breakout magnitude trend (3-day moving average of breakout strength)
    data['up_breakout_momentum'] = data['up_breakout_strength'].rolling(window=3).mean()
    data['down_breakout_momentum'] = data['down_breakout_strength'].rolling(window=3).mean()
    
    # Volume-Price Confirmation
    # Volume breakout alignment
    data['up_breakout_volume'] = np.where(
        data['up_breakout'] == 1,
        data['volume'],
        0
    )
    data['down_breakout_volume'] = np.where(
        data['down_breakout'] == 1,
        data['volume'],
        0
    )
    
    # 5-day volume ratios
    data['up_volume_5'] = data['up_breakout_volume'].rolling(window=5).sum()
    data['down_volume_5'] = data['down_breakout_volume'].rolling(window=5).sum()
    data['volume_ratio'] = data['up_volume_5'] / (data['down_volume_5'] + 1e-8)
    
    # Intraday range confirmation
    data['range_vs_avg'] = data['daily_range'] / (data['avg_range_5'] + 1e-8)
    data['close_position'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8)
    
    # Breakout confirmation strength
    data['up_breakout_confirmed'] = np.where(
        (data['up_breakout'] == 1) & (data['close_position'] > 0.6) & (data['range_vs_avg'] > 1),
        data['up_breakout_strength'] * data['volume_ratio'],
        0
    )
    data['down_breakout_confirmed'] = np.where(
        (data['down_breakout'] == 1) & (data['close_position'] < 0.4) & (data['range_vs_avg'] > 1),
        data['down_breakout_strength'] / (data['volume_ratio'] + 1e-8),
        0
    )
    
    # Generate Adaptive Alpha Signal
    # Volume-weighted breakout asymmetry
    data['volume_weighted_asymmetry'] = (
        data['net_breakout_ratio'] * np.log1p(data['volume_ratio'])
    )
    
    # Breakout persistence momentum
    data['breakout_persistence'] = (
        data['up_streak'] * data['up_breakout_momentum'] - 
        data['down_streak'] * data['down_breakout_momentum']
    )
    
    # Confirmed directional signal strength
    data['confirmed_signal'] = (
        data['up_breakout_confirmed'].rolling(window=3).mean() - 
        data['down_breakout_confirmed'].rolling(window=3).mean()
    )
    
    # Final alpha factor combining all components
    alpha = (
        0.4 * data['volume_weighted_asymmetry'] +
        0.3 * data['breakout_persistence'] +
        0.3 * data['confirmed_signal']
    )
    
    # Normalize the alpha signal
    alpha_std = alpha.rolling(window=20).std()
    alpha = alpha / (alpha_std + 1e-8)
    
    return alpha
