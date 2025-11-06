import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Robust momentum using median-based price change (5-day window)
    price_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Volume acceleration using robust rolling median (10-day window)
    volume_median = df['volume'].rolling(window=10, min_periods=5).median()
    volume_acceleration = (df['volume'] - volume_median) / (volume_median + 1e-7)
    
    # Robust volatility estimation using median absolute deviation (10-day window)
    price_range = (df['high'] - df['low']) / df['close']
    volatility_mad = price_range.rolling(window=10, min_periods=5).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    
    # Clip extreme values to reduce outlier influence
    price_momentum_clipped = price_momentum.clip(lower=price_momentum.quantile(0.05), 
                                               upper=price_momentum.quantile(0.95))
    volume_acceleration_clipped = volume_acceleration.clip(lower=volume_acceleration.quantile(0.05), 
                                                         upper=volume_acceleration.quantile(0.95))
    
    # Multiplicative interaction: volatility-normalized momentum × volume confirmation
    # The factor amplifies momentum signals that are confirmed by volume acceleration
    volatility_normalized_momentum = price_momentum_clipped / (volatility_mad + 1e-7)
    volume_confirmation = 1 + np.tanh(volume_acceleration_clipped)  # Smooth volume amplification
    
    alpha_factor = volatility_normalized_momentum * volume_confirmation
    
    return alpha_factor
