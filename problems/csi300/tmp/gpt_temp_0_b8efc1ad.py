import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-efficiency momentum factor
    # Combines short-term momentum with volatility normalization and volume efficiency
    # Higher values indicate strong momentum in efficient trading conditions
    
    # 1. Short-term volatility-normalized momentum (3-day momentum)
    momentum = (df['close'] - df['close'].shift(3)) / ((df['high'] - df['low']).rolling(5).std() + 1e-7)
    
    # 2. Volume efficiency ratio (price change per unit volume)
    volume_efficiency = (df['close'] - df['close'].shift(1)).abs() / (df['volume'] + 1e-7)
    volume_efficiency_rank = volume_efficiency.rolling(5).apply(lambda x: x.rank(pct=True).iloc[-1])
    
    # 3. Price range efficiency (close position within daily range)
    range_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    range_efficiency = 1 - (range_position - 0.5).abs() * 2
    
    # 4. Volume trend confirmation (volume acceleration with price direction)
    volume_acceleration = df['volume'] / df['volume'].rolling(3).mean()
    direction_confirmation = volume_acceleration * np.sign(df['close'] - df['close'].shift(1))
    
    # Combine components focusing on momentum strength and efficiency
    alpha_factor = (
        momentum * 0.40 +
        volume_efficiency_rank * 0.25 +
        range_efficiency * 0.20 +
        direction_confirmation * 0.15
    )
    
    return alpha_factor
