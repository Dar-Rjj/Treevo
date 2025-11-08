import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Core Price Components
    df['intraday_momentum'] = df['close'] - df['open']
    df['daily_range'] = df['high'] - df['low']
    df['close_lag1'] = df['close'].shift(1)
    
    # Multi-Timeframe Momentum
    # Ultra-Short (1-2 days)
    df['momentum_1d'] = df['close'] - df['close_lag1']
    df['range_1d'] = df['daily_range']
    
    # Short-Term (3-5 days)
    df['close_lag2'] = df['close'].shift(2)
    df['momentum_3d'] = df['close'] - df['close_lag2']
    df['range_3d'] = df['daily_range'] + df['daily_range'].shift(1) + df['daily_range'].shift(2)
    
    # Medium-Term (6-10 days)
    df['close_lag9'] = df['close'].shift(9)
    df['momentum_10d'] = df['close'] - df['close_lag9']
    df['range_10d'] = df['daily_range'].rolling(window=10).sum()
    
    # Volatility Regime Framework
    df['short_term_vol'] = df['range_3d'] / 3
    df['medium_term_vol'] = df['range_10d'] / 10
    df['volatility_ratio'] = df['short_term_vol'] / df['medium_term_vol']
    
    # Regime Classification
    conditions = [
        df['volatility_ratio'] > 1.15,
        (df['volatility_ratio'] >= 0.85) & (df['volatility_ratio'] <= 1.15),
        df['volatility_ratio'] < 0.85
    ]
    choices = ['high_vol', 'normal_vol', 'low_vol']
    df['vol_regime'] = np.select(conditions, choices, default='normal_vol')
    
    # Volume Persistence Engine
    df['volume_lag1'] = df['volume'].shift(1)
    df['volume_change'] = df['volume'] - df['volume_lag1']
    df['volume_direction'] = np.sign(df['volume_change'])
    
    # Volume Streak calculation
    df['volume_streak'] = 0
    for i in range(1, len(df)):
        if df['volume_direction'].iloc[i] == df['volume_direction'].iloc[i-1]:
            df['volume_streak'].iloc[i] = df['volume_streak'].iloc[i-1] + 1
        else:
            df['volume_streak'].iloc[i] = 1
    
    # Volume-Momentum Alignment
    df['alignment'] = np.sign(df['momentum_1d']) * np.sign(df['volume_change'])
    df['alignment_streak'] = 0
    for i in range(1, len(df)):
        if df['alignment'].iloc[i] > 0:
            df['alignment_streak'].iloc[i] = df['alignment_streak'].iloc[i-1] + 1
        else:
            df['alignment_streak'].iloc[i] = 0
    
    df['alignment_strength'] = df['alignment_streak'] * np.abs(df['momentum_1d'])
    
    # Volume Regime
    df['volume_3d'] = df['volume'].rolling(window=3).mean()
    df['volume_10d'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume_3d'] / df['volume_10d']
    
    vol_conditions = [
        df['volume_ratio'] > 1.1,
        (df['volume_ratio'] >= 0.9) & (df['volume_ratio'] <= 1.1),
        df['volume_ratio'] < 0.9
    ]
    vol_choices = ['high_volume', 'normal_volume', 'low_volume']
    df['volume_regime'] = np.select(vol_conditions, vol_choices, default='normal_volume')
    
    # Adaptive Factor Construction
    # Volatility-Scaled Momentum
    df['vsm_1d'] = df['momentum_1d'] / df['range_1d']
    df['vsm_3d'] = df['momentum_3d'] / df['range_3d']
    df['vsm_10d'] = df['momentum_10d'] / df['range_10d']
    
    # Multi-Timeframe Blend with Regime-Based Weights
    base_signal = pd.Series(index=df.index, dtype=float)
    
    # High Vol regime
    high_vol_mask = df['vol_regime'] == 'high_vol'
    base_signal[high_vol_mask] = (6 * df['vsm_1d'] + 2 * df['vsm_3d']) / 8
    
    # Normal Vol regime
    normal_vol_mask = df['vol_regime'] == 'normal_vol'
    base_signal[normal_vol_mask] = (4 * df['vsm_1d'] + 3 * df['vsm_3d'] + df['vsm_10d']) / 8
    
    # Low Vol regime
    low_vol_mask = df['vol_regime'] == 'low_vol'
    base_signal[low_vol_mask] = (2 * df['vsm_1d'] + 3 * df['vsm_3d'] + 3 * df['vsm_10d']) / 8
    
    # Volume Integration
    base_signal = base_signal * np.log(df['volume'] + 1)
    
    # Volume Persistence Enhancement
    # Volume Confirmation
    volume_confirmed_signal = base_signal.copy()
    positive_alignment_mask = df['alignment'] > 0
    negative_alignment_mask = df['alignment'] < 0
    
    volume_confirmed_signal[positive_alignment_mask] *= (1 + df['alignment_streak'] / 5)
    volume_confirmed_signal[negative_alignment_mask] *= (1 - df['alignment_streak'] / 5)
    
    # Volume Regime Scaling
    high_volume_mask = df['volume_regime'] == 'high_volume'
    normal_volume_mask = df['volume_regime'] == 'normal_volume'
    low_volume_mask = df['volume_regime'] == 'low_volume'
    
    volume_confirmed_signal[high_volume_mask] *= 1.2
    volume_confirmed_signal[normal_volume_mask] *= 1.0
    volume_confirmed_signal[low_volume_mask] *= 0.8
    
    # Momentum Acceleration
    df['acceleration'] = df['vsm_3d'] - df['vsm_10d']
    df['acceleration_direction'] = np.sign(df['acceleration'])
    
    final_signal = volume_confirmed_signal * (1 + 0.1 * df['acceleration_direction'])
    
    return final_signal
