import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate directional consistency across timeframes
    df = df.copy()
    
    # 3-day, 8-day, and 15-day price directions
    df['price_dir_3d'] = np.sign(df['close'] - df['close'].shift(3))
    df['price_dir_8d'] = np.sign(df['close'] - df['close'].shift(8))
    df['price_dir_15d'] = np.sign(df['close'] - df['close'].shift(15))
    
    # Directional agreement score (percentage of matching signs)
    dir_cols = ['price_dir_3d', 'price_dir_8d', 'price_dir_15d']
    df['dir_agreement'] = df[dir_cols].apply(
        lambda x: (x == x.mode()[0]).sum() / len(x) if len(x.mode()) == 1 else 0.33, axis=1
    )
    
    # Volume participation patterns
    df['volume_intensity'] = df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
    df['volume_intensity_5d_trend'] = df['volume_intensity'].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
    )
    
    # Volume divergence from price direction
    df['volume_divergence'] = (
        np.sign(df['volume_intensity_5d_trend']) != np.sign(df['price_dir_3d'])
    ).astype(float)
    
    # Price movement quality metrics
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['intraday_capture_ratio'] = (
        (df['close'] - df['close'].shift(1)) / df['true_range']
    ).replace([np.inf, -np.inf], 0)
    
    # Price persistence (streak of same-direction daily closes)
    df['daily_return'] = df['close'].pct_change()
    df['direction'] = np.sign(df['daily_return'])
    streak = []
    current_streak = 0
    current_dir = 0
    for i, dir_val in enumerate(df['direction']):
        if np.isnan(dir_val):
            streak.append(0)
            continue
        if dir_val == current_dir:
            current_streak += 1
        else:
            current_streak = 1 if dir_val != 0 else 0
            current_dir = dir_val
        streak.append(current_streak)
    df['price_persistence'] = streak
    
    # Gap behavior
    df['gap_ratio'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Closing pressure (approximated by close relative to day's range)
    df['closing_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Generate divergence-based alpha signal
    df['time_decay_weight'] = np.exp(-0.1 * df['price_persistence'])
    
    # Combine components with weights and filters
    alpha_signal = (
        df['dir_agreement'] * 
        (1 - df['volume_divergence']) *  # Penalize volume divergence
        df['intraday_capture_ratio'].abs() *  # Weight by capture efficiency
        df['price_persistence'] *  # Reward persistent moves
        df['time_decay_weight'] *  # Apply time decay
        np.where(df['volume_intensity'] > df['volume_intensity'].rolling(20).median(), 1.5, 0.8)  # Volume filter
    )
    
    # Final smoothing and normalization
    alpha_signal = alpha_signal.rolling(3).mean()
    alpha_signal = (alpha_signal - alpha_signal.rolling(50).mean()) / alpha_signal.rolling(50).std()
    
    return alpha_signal
