import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Core Price Components
    df['intraday_momentum'] = df['close'] - df['open']
    df['daily_range'] = df['high'] - df['low']
    df['prev_close'] = df['close'].shift(1)
    
    # Multi-Timeframe Momentum
    # Short-term (3-day)
    df['price_change_3d'] = df['close'] - df['close'].shift(2)
    df['avg_range_3d'] = (df['daily_range'] + df['daily_range'].shift(1) + df['daily_range'].shift(2)) / 3
    
    # Medium-term (10-day)
    df['price_change_10d'] = df['close'] - df['close'].shift(9)
    df['avg_range_10d'] = df['daily_range'].rolling(window=10, min_periods=10).mean()
    
    # Volatility Scaling
    df['vol_ratio'] = df['avg_range_3d'] / df['avg_range_10d']
    
    # Volatility-Scaled Momentum
    df['VSM_3d'] = df['price_change_3d'] / df['avg_range_3d']
    df['VSM_10d'] = df['price_change_10d'] / df['avg_range_10d']
    
    # Volume Analysis
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_direction'] = np.sign(df['volume_change'])
    
    # Volume Persistence
    df['consecutive_days'] = 0
    df['persistence_strength'] = 0
    
    # Calculate consecutive days with same volume direction
    for i in range(1, len(df)):
        if df['volume_direction'].iloc[i] == df['volume_direction'].iloc[i-1]:
            df.loc[df.index[i], 'consecutive_days'] = df['consecutive_days'].iloc[i-1] + 1
        else:
            df.loc[df.index[i], 'consecutive_days'] = 1
    
    df['persistence_strength'] = df['consecutive_days'] * abs(df['volume_change'])
    
    # Volume-Momentum Alignment
    df['direction_match'] = np.sign(df['intraday_momentum']) * df['volume_direction']
    df['alignment_streak'] = 0
    df['alignment_confidence'] = 0
    
    # Calculate alignment streak
    for i in range(1, len(df)):
        if df['direction_match'].iloc[i] > 0:
            df.loc[df.index[i], 'alignment_streak'] = df['alignment_streak'].iloc[i-1] + 1
        else:
            df.loc[df.index[i], 'alignment_streak'] = 0
    
    df['alignment_confidence'] = df['alignment_streak'] * abs(df['intraday_momentum'])
    
    # Factor Construction
    # Base Momentum Signal
    df['weighted_VSM'] = (2 * df['VSM_3d'] + df['VSM_10d']) / 3
    df['volume_integrated'] = df['weighted_VSM'] * np.log(df['volume'] + 1)
    
    # Volatility Adjustment
    df['volatility_adjusted'] = df['volume_integrated'].copy()
    high_vol_mask = df['vol_ratio'] > 1.2
    normal_vol_mask = (df['vol_ratio'] >= 0.8) & (df['vol_ratio'] <= 1.2)
    low_vol_mask = df['vol_ratio'] < 0.8
    
    df.loc[high_vol_mask, 'volatility_adjusted'] = df.loc[high_vol_mask, 'volume_integrated'] * 0.7
    df.loc[normal_vol_mask, 'volatility_adjusted'] = df.loc[normal_vol_mask, 'volume_integrated'] * 1.0
    df.loc[low_vol_mask, 'volatility_adjusted'] = df.loc[low_vol_mask, 'volume_integrated'] * 1.3
    
    # Volume Confirmation
    df['volume_confirmed'] = df['volatility_adjusted'].copy()
    positive_alignment_mask = df['direction_match'] > 0
    negative_alignment_mask = df['direction_match'] < 0
    
    df.loc[positive_alignment_mask, 'volume_confirmed'] = df.loc[positive_alignment_mask, 'volatility_adjusted'] * (1 + df.loc[positive_alignment_mask, 'alignment_streak'] / 5)
    df.loc[negative_alignment_mask, 'volume_confirmed'] = df.loc[negative_alignment_mask, 'volatility_adjusted'] * (1 - df.loc[negative_alignment_mask, 'alignment_streak'] / 5)
    
    # Persistence Enhancement
    df['persistence_multiplier'] = 1 + (df['consecutive_days'] / 10)
    df['final_factor'] = df['volume_confirmed'] * df['persistence_multiplier']
    
    return df['final_factor']
