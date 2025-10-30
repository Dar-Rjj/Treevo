import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volatility-Regime Volume Acceleration Mean Reversion factor
    """
    df = data.copy()
    
    # Volatility Regime Classification
    # Daily volatility proxy using high-low range
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Dynamic volatility baseline with exponential weighting
    df['vol_baseline'] = df['daily_range'].rolling(window=60, min_periods=20).median()
    # Apply exponential decay smoothing (λ=0.94)
    alpha = 0.06  # 1 - 0.94
    df['vol_baseline_smooth'] = df['vol_baseline'].ewm(alpha=alpha, adjust=False).mean()
    
    # Regime classification with hysteresis
    df['vol_regime'] = 'normal'
    high_vol_threshold = 1.5 * df['vol_baseline_smooth']
    low_vol_threshold = 0.8 * df['vol_baseline_smooth']
    
    # Apply hysteresis: require 2 consecutive days to change regime
    for i in range(2, len(df)):
        current_range = df['daily_range'].iloc[i]
        prev_regime = df['vol_regime'].iloc[i-1]
        
        if prev_regime == 'high':
            if current_range <= 1.3 * df['vol_baseline_smooth'].iloc[i]:
                df.loc[df.index[i], 'vol_regime'] = 'normal'
            else:
                df.loc[df.index[i], 'vol_regime'] = 'high'
        elif prev_regime == 'low':
            if current_range >= 0.9 * df['vol_baseline_smooth'].iloc[i]:
                df.loc[df.index[i], 'vol_regime'] = 'normal'
            else:
                df.loc[df.index[i], 'vol_regime'] = 'low'
        else:  # normal regime
            if current_range > high_vol_threshold.iloc[i]:
                df.loc[df.index[i], 'vol_regime'] = 'high'
            elif current_range < low_vol_threshold.iloc[i]:
                df.loc[df.index[i], 'vol_regime'] = 'low'
            else:
                df.loc[df.index[i], 'vol_regime'] = 'normal'
    
    # Volume Acceleration Measurement
    df['volume_ma3'] = df['volume'].rolling(window=3, min_periods=2).mean()
    df['volume_ma8'] = df['volume'].rolling(window=8, min_periods=5).mean()
    df['volume_ma5'] = df['volume'].rolling(window=5, min_periods=3).mean()
    df['volume_ma15'] = df['volume'].rolling(window=15, min_periods=10).mean()
    
    df['vol_ratio_3_8'] = df['volume_ma3'] / df['volume_ma8']
    df['vol_ratio_5_15'] = df['volume_ma5'] / df['volume_ma15']
    
    # Volume acceleration classification
    df['vol_accel'] = 'none'
    strong_condition = (df['vol_ratio_3_8'] > 1.25) & (df['vol_ratio_5_15'] > 1.25)
    moderate_condition = (df['vol_ratio_3_8'] > 1.15) | (df['vol_ratio_5_15'] > 1.15)
    
    df.loc[strong_condition, 'vol_accel'] = 'strong'
    df.loc[~strong_condition & moderate_condition, 'vol_accel'] = 'moderate'
    
    # Price Reversion Signal
    df['return_3d'] = df['close'] / df['close'].shift(3) - 1
    df['vol_adjusted_return'] = df['return_3d'] / (df['daily_range'] + 0.01)
    
    # Cubic transformation for non-linear scaling
    df['reversion_signal'] = -df['vol_adjusted_return'] ** 3
    
    # Signal Integration with Regime-Volume Interaction
    df['final_signal'] = 0.0
    
    # High Volatility + Strong Volume Acceleration
    high_strong_mask = (df['vol_regime'] == 'high') & (df['vol_accel'] == 'strong')
    vol_magnitude = (df['vol_ratio_3_8'] + df['vol_ratio_5_15']) / 2
    df.loc[high_strong_mask, 'final_signal'] = 2.5 * df['reversion_signal'] + 0.1 * vol_magnitude
    
    # High Volatility + Moderate Volume Acceleration
    high_moderate_mask = (df['vol_regime'] == 'high') & (df['vol_accel'] == 'moderate')
    vol_avg = (df['vol_ratio_3_8'] + df['vol_ratio_5_15']) / 2
    df.loc[high_moderate_mask, 'final_signal'] = 1.8 * df['reversion_signal'] * vol_avg
    
    # Normal Volatility + Strong Acceleration
    normal_strong_mask = (df['vol_regime'] == 'normal') & (df['vol_accel'] == 'strong')
    regime_strength = (df['daily_range'] / df['vol_baseline_smooth'] - 1).clip(-0.5, 0.5)
    df.loc[normal_strong_mask, 'final_signal'] = 1.5 * df['reversion_signal'] * (1 + regime_strength)
    
    # Normal Volatility + Moderate Acceleration
    normal_moderate_mask = (df['vol_regime'] == 'normal') & (df['vol_accel'] == 'moderate')
    df.loc[normal_moderate_mask, 'final_signal'] = 1.2 * df['reversion_signal']
    
    # Low Volatility Regime
    low_vol_mask = (df['vol_regime'] == 'low')
    df.loc[low_vol_mask, 'final_signal'] = 0.3 * df['reversion_signal']
    
    # Fill NaN values and return
    result = df['final_signal'].fillna(0)
    return result
