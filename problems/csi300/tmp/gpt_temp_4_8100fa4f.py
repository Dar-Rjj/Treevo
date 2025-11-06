import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 3-5 day momentum acceleration: difference between 3-day and 5-day momentum
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_acceleration = momentum_3d - momentum_5d
    
    # 3-5 day volume-amount alignment: correlation between volume and amount changes
    volume_change_3d = df['volume'] / df['volume'].shift(3) - 1
    amount_change_3d = df['amount'] / df['amount'].shift(3) - 1
    volume_change_5d = df['volume'] / df['volume'].shift(5) - 1
    amount_change_5d = df['amount'] / df['amount'].shift(5) - 1
    
    # Simple alignment measure: product of 3-day and 5-day changes
    volume_amount_alignment = (volume_change_3d * amount_change_3d + 
                              volume_change_5d * amount_change_5d)
    
    # Blend momentum acceleration with volume-amount alignment
    blended_factor = momentum_acceleration * volume_amount_alignment
    
    # 10-day volatility using daily range
    daily_range = (df['high'] - df['low']) / df['close']
    volatility_10d = daily_range.rolling(window=10).std()
    
    # Normalize by 10-day volatility
    normalized_factor = blended_factor / (volatility_10d + 1e-7)
    
    # Apply exponential decay with 5-day half-life
    decay_factor = 0.5 ** (1/5)  # 5-day half-life
    weights = [decay_factor ** i for i in range(len(normalized_factor))]
    weights = pd.Series(weights, index=normalized_factor.index)[::-1]
    
    # Apply weighted average favoring recent data
    decayed_factor = normalized_factor.rolling(window=len(normalized_factor), 
                                              min_periods=1).apply(
        lambda x: np.average(x, weights=weights[-len(x):])
    )
    
    return decayed_factor
