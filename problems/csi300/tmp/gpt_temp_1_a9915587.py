import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Robust momentum using median-based price change over 3 days
    price_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Volume acceleration using rolling median to reduce noise
    volume_median = df['volume'].rolling(window=5, min_periods=3).median()
    volume_acceleration = (df['volume'] - volume_median) / (volume_median + 1e-7)
    
    # Robust volatility using median absolute deviation of daily range
    daily_range = df['high'] - df['low']
    vol_robust = daily_range.rolling(window=10, min_periods=7).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    
    # Clip extreme values to handle outliers (95th percentile winsorization)
    price_momentum_clipped = price_momentum.clip(
        lower=price_momentum.quantile(0.025),
        upper=price_momentum.quantile(0.975)
    )
    volume_acceleration_clipped = volume_acceleration.clip(
        lower=volume_acceleration.quantile(0.025),
        upper=volume_acceleration.quantile(0.975)
    )
    
    # Multiplicative interaction: volatility-normalized momentum × volume confirmation
    # Volume term uses sigmoid-like transformation for intuitive amplification
    volatility_normalized_momentum = price_momentum_clipped / (vol_robust + 1e-7)
    volume_confirmation = 1 + np.tanh(volume_acceleration_clipped * 2)
    
    # Final alpha factor with economic interpretation:
    # Strong momentum signals are amplified when supported by volume acceleration
    # but dampened during high volatility periods
    alpha_factor = volatility_normalized_momentum * volume_confirmation
    
    return alpha_factor
