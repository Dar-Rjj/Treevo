import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Dynamic momentum blending (2-day and 5-day)
    momentum_2d = (df['close'] / df['close'].shift(2)) - 1
    momentum_5d = (df['close'] / df['close'].shift(5)) - 1
    blended_momentum = momentum_2d * momentum_5d
    
    # Volume trend across multiple timeframes
    volume_ratio_3d = df['volume'] / df['volume'].rolling(window=3).mean()
    volume_ratio_8d = df['volume'] / df['volume'].rolling(window=8).mean()
    volume_trend = volume_ratio_3d * volume_ratio_8d
    
    # Robust volatility using median absolute deviation
    returns_5d = df['close'].pct_change(periods=5)
    volatility_5d = returns_5d.rolling(window=5).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    
    # Combined factor with multiplicative interactions
    alpha_factor = blended_momentum * volume_trend / (volatility_5d + 1e-7)
    
    return alpha_factor
