import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Convergence Factor
    Combines short-term (1-3 days) and medium-term (5-10 days) momentum signals
    with volume confirmation across multiple timeframes
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Short-term momentum calculations (1-3 days)
    # Price-based short-term momentum
    data['ret_1d'] = data['close'].pct_change(1)
    data['ret_3d'] = data['close'].pct_change(3)
    
    # Volume-based short-term momentum
    data['vol_ma_3d'] = data['volume'].rolling(window=3, min_periods=1).mean()
    data['vol_ratio_3d'] = data['volume'] / data['vol_ma_3d'].shift(1)
    data['vol_accel_3d'] = (data['volume'] / data['volume'].shift(1)) / (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Medium-term momentum calculations (5-10 days)
    # Price-based medium-term momentum
    data['ret_5d'] = data['close'].pct_change(5)
    data['ret_10d'] = data['close'].pct_change(10)
    
    # Volume-based medium-term momentum
    data['vol_ma_10d'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['vol_ratio_10d'] = data['volume'] / data['vol_ma_10d'].shift(1)
    data['vol_trend_10d'] = data['volume'].rolling(window=10, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan
    )
    
    # Momentum alignment across timeframes
    # Sign agreement: count how many momentum signals have the same direction
    data['momentum_sign_agreement'] = (
        (data['ret_1d'] > 0).astype(int) + 
        (data['ret_3d'] > 0).astype(int) + 
        (data['ret_5d'] > 0).astype(int) + 
        (data['ret_10d'] > 0).astype(int)
    )
    
    # Magnitude consistency: normalized momentum strength
    momentum_signals = ['ret_1d', 'ret_3d', 'ret_5d', 'ret_10d']
    data['momentum_magnitude_std'] = data[momentum_signals].std(axis=1)
    data['momentum_magnitude_mean'] = data[momentum_signals].mean(axis=1)
    data['momentum_consistency'] = data['momentum_magnitude_mean'] / (data['momentum_magnitude_std'] + 1e-8)
    
    # Volume confirmation across timeframes
    volume_signals = ['vol_ratio_3d', 'vol_accel_3d', 'vol_ratio_10d']
    data['volume_support'] = (
        (data['vol_ratio_3d'] > 1).astype(int) + 
        (data['vol_accel_3d'] > 1).astype(int) + 
        (data['vol_ratio_10d'] > 1).astype(int)
    )
    
    # Volume acceleration alignment
    data['volume_accel_alignment'] = (
        np.sign(data['vol_accel_3d']) * np.sign(data['vol_trend_10d'])
    ).fillna(0)
    
    # Combined signal strength
    # Weight short-term momentum more heavily
    short_term_weight = 0.6
    medium_term_weight = 0.4
    
    # Normalize momentum signals
    for col in momentum_signals:
        data[f'{col}_norm'] = data[col] / (data[col].rolling(window=20, min_periods=1).std() + 1e-8)
    
    # Calculate weighted momentum convergence
    data['momentum_convergence'] = (
        short_term_weight * (data['ret_1d_norm'] + data['ret_3d_norm']) / 2 +
        medium_term_weight * (data['ret_5d_norm'] + data['ret_10d_norm']) / 2
    )
    
    # Apply volume confirmation
    volume_strength = data['volume_support'] / 3.0  # Normalize to 0-1
    volume_alignment_strength = (data['volume_accel_alignment'] + 1) / 2  # Convert -1,1 to 0,1
    
    # Final factor: momentum convergence enhanced by volume confirmation
    data['factor'] = (
        data['momentum_convergence'] * 
        (0.6 + 0.4 * volume_strength) *  # Volume support adds up to 40% enhancement
        (0.7 + 0.3 * volume_alignment_strength)  # Volume alignment adds up to 30% enhancement
    )
    
    # Apply momentum sign agreement filter
    # Require at least 2 out of 4 momentum signals to agree in direction
    data['factor'] = data['factor'] * (data['momentum_sign_agreement'] >= 2)
    
    # Apply momentum consistency filter (avoid conflicting signals)
    data['factor'] = data['factor'] * np.exp(-data['momentum_magnitude_std'])
    
    return data['factor']
