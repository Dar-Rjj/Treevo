import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Normalized Multi-Timeframe Momentum with Volume Persistence
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate true range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Multi-Timeframe Momentum Calculation
    # Short-term (1-3 days)
    df['mom_short_raw'] = (df['close'] - df['close'].shift(2)) / (
        df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min() + 1e-8)
    
    # Medium-term (5-10 days)
    df['mom_medium_raw'] = (df['close'] - df['close'].shift(5)) / (
        df['high'].rolling(window=6).max() - df['low'].rolling(window=6).min() + 1e-8)
    
    # Long-term (20 days)
    df['mom_long_raw'] = (df['close'] - df['close'].shift(20)) / (
        df['high'].rolling(window=21).max() - df['low'].rolling(window=21).min() + 1e-8)
    
    # Momentum Persistence Analysis
    # Directional consistency
    df['sign_short'] = np.sign(df['mom_short_raw'])
    df['sign_medium'] = np.sign(df['mom_medium_raw'])
    df['sign_long'] = np.sign(df['mom_long_raw'])
    
    # Count matching signs with timeframe weights
    df['dir_persistence'] = (
        (df['sign_short'] == df['sign_medium']).astype(int) * 0.5 +
        (df['sign_short'] == df['sign_long']).astype(int) * 0.3 +
        (df['sign_medium'] == df['sign_long']).astype(int) * 0.2
    )
    
    # Strength persistence tracking
    mom_strength_threshold = df['true_range'].rolling(window=10).std() * 0.5
    df['strong_mom'] = (abs(df['mom_short_raw']) > mom_strength_threshold).astype(int)
    
    # Track consecutive strong momentum days with decay
    df['strength_persistence'] = 0
    for i in range(1, len(df)):
        if df['strong_mom'].iloc[i]:
            decay = 0.95 if df['true_range'].iloc[i] > df['true_range'].iloc[i-1] else 0.98
            df['strength_persistence'].iloc[i] = df['strength_persistence'].iloc[i-1] * decay + 1
        else:
            df['strength_persistence'].iloc[i] = 0
    
    # Combine directional and strength persistence
    df['momentum_persistence'] = df['dir_persistence'] * (1 + df['strength_persistence'] / 10)
    
    # Volume Persistence Confirmation
    # Volume trends
    df['vol_trend_short'] = df['volume'] / (df['volume'].shift(3) + 1e-8)
    df['vol_trend_medium'] = df['volume'] / (df['volume'].shift(10) + 1e-8)
    
    # Volume-momentum alignment
    df['vol_mom_aligned'] = (
        (np.sign(df['vol_trend_short'] - 1) == df['sign_short']).astype(int) * 0.6 +
        (np.sign(df['vol_trend_medium'] - 1) == df['sign_medium']).astype(int) * 0.4
    )
    
    # Track consecutive alignment days
    df['vol_persistence'] = 0
    for i in range(1, len(df)):
        if df['vol_mom_aligned'].iloc[i] > 0.5:
            df['vol_persistence'].iloc[i] = df['vol_persistence'].iloc[i-1] + 1
        else:
            df['vol_persistence'].iloc[i] = 0
    
    # Volume confirmation score
    vol_strength = (abs(df['vol_trend_short'] - 1) + abs(df['vol_trend_medium'] - 1)) / 2
    df['volume_confirmation'] = df['vol_persistence'] * vol_strength / 10
    
    # Adaptive Factor Construction
    # Weighted momentum average
    momentum_weights = [0.5, 0.3, 0.2]  # short, medium, long
    df['weighted_momentum'] = (
        df['mom_short_raw'] * momentum_weights[0] +
        df['mom_medium_raw'] * momentum_weights[1] +
        df['mom_long_raw'] * momentum_weights[2]
    )
    
    # Apply persistence multipliers
    df['persistence_adjusted'] = df['weighted_momentum'] * (1 + df['momentum_persistence'])
    
    # Volume confirmation adjustment
    df['volume_adjusted'] = df['persistence_adjusted'] * (1 + df['volume_confirmation'])
    
    # Volatility-adaptive scaling
    recent_volatility = df['true_range'].rolling(window=5).mean()
    volatility_scaling = 1 / (recent_volatility + 1e-8)
    
    # Final factor with volatility adjustment
    alpha_factor = df['volume_adjusted'] * volatility_scaling
    
    # Clean up intermediate columns
    result = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result
