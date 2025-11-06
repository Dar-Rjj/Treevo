import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-resolution momentum factor with adaptive volatility scaling and 
    volume-amount divergence detection.
    
    Economic intuition:
    - Multi-resolution momentum captures trends across nested time horizons
    - Rolling average references provide stable momentum baselines
    - Volume-amount divergence detects unusual trading patterns
    - Adaptive volatility scaling adjusts for changing market conditions
    - Composite confirmation strengthens signal reliability
    """
    
    # Multi-resolution momentum using nested rolling averages
    # Ultra-short momentum (2-day) vs 5-day baseline
    close_avg_5d = df['close'].rolling(window=5, min_periods=5).mean()
    momentum_2d = (df['close'] - close_avg_5d.shift(2)) / close_avg_5d.shift(2)
    
    # Short-term momentum (5-day) vs 10-day baseline
    close_avg_10d = df['close'].rolling(window=10, min_periods=10).mean()
    momentum_5d = (df['close'] - close_avg_10d.shift(5)) / close_avg_10d.shift(5)
    
    # Medium-term momentum (10-day) vs 20-day baseline
    close_avg_20d = df['close'].rolling(window=20, min_periods=20).mean()
    momentum_10d = (df['close'] - close_avg_20d.shift(10)) / close_avg_20d.shift(10)
    
    # Long-term momentum (20-day) vs 40-day baseline
    close_avg_40d = df['close'].rolling(window=40, min_periods=40).mean()
    momentum_20d = (df['close'] - close_avg_40d.shift(20)) / close_avg_40d.shift(20)
    
    # Volume-amount divergence detection
    # Normalized volume and amount using rolling percentiles
    volume_rank_10d = df['volume'].rolling(window=10, min_periods=10).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-7)
    )
    amount_rank_10d = df['amount'].rolling(window=10, min_periods=10).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-7)
    )
    
    # Divergence measure: unusual volume-amount patterns
    volume_amount_divergence = abs(volume_rank_10d - amount_rank_10d)
    
    # Composite volume-amount strength with divergence penalty
    volume_strength = df['volume'] / df['volume'].rolling(window=10, min_periods=10).mean()
    amount_strength = df['amount'] / df['amount'].rolling(window=10, min_periods=10).mean()
    composite_strength = (volume_strength + amount_strength) / 2
    
    # Apply divergence penalty: reduce strength when patterns are unusual
    adjusted_strength = composite_strength / (1 + volume_amount_divergence)
    
    # Adaptive volatility scaling using multi-timeframe true range
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    
    # Multi-resolution volatility estimates
    vol_short = true_range.rolling(window=5, min_periods=5).mean()
    vol_medium = true_range.rolling(window=10, min_periods=10).mean()
    vol_long = true_range.rolling(window=20, min_periods=20).mean()
    
    # Adaptive volatility scaling: blend based on recent volatility changes
    vol_change_ratio = vol_short / vol_medium
    adaptive_volatility = np.where(
        vol_change_ratio > 1.2,
        vol_short,  # Use short-term vol during high volatility periods
        np.where(
            vol_change_ratio < 0.8,
            vol_long,  # Use long-term vol during low volatility periods
            vol_medium  # Default to medium-term vol
        )
    )
    
    # Multi-resolution momentum with exponential time decay
    multi_res_momentum = (
        0.5 * momentum_2d +
        0.3 * momentum_5d +
        0.15 * momentum_10d +
        0.05 * momentum_20d
    )
    
    # Final factor: multi-resolution momentum enhanced by adjusted strength
    # and scaled by adaptive volatility
    factor = (multi_res_momentum * adjusted_strength) / (adaptive_volatility + 1e-7)
    
    return factor
