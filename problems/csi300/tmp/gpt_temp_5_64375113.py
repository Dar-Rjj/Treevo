import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Volatility-normalized momentum-volume alignment factor.
    Combines medium-term momentum with volume trends, normalized by volatility,
    using decay weights for robustness and directional stability.
    """
    # Medium-term momentum (5-day for stability)
    momentum = df['close'] / df['close'].shift(5) - 1
    
    # Volume trend alignment (5-day volume vs 20-day average)
    volume_trend = df['volume'] / df['volume'].rolling(window=20).mean() - 1
    
    # Volatility normalization (20-day rolling std)
    volatility = df['close'].pct_change().rolling(window=20).std()
    
    # Multiplicative combination with alignment penalty
    # Positive when momentum and volume trend align, negative when they diverge
    raw_factor = momentum * volume_trend
    
    # Volatility normalization (avoid division by zero)
    volatility_normalized = raw_factor / (volatility + 1e-7)
    
    # Apply exponential decay weights (0.95 decay over 5 days)
    weights = 0.95 ** np.arange(5)
    decayed_factor = volatility_normalized.rolling(window=5).apply(
        lambda x: np.sum(x * weights) / np.sum(weights) if len(x) == 5 else np.nan
    )
    
    # Directional stability filter (sign consistency over 3 days)
    sign_consistency = decayed_factor.rolling(window=3).apply(
        lambda x: 1 if len(x) == 3 and np.all(np.sign(x) == np.sign(x[0])) else 0
    )
    
    # Final factor with stability enhancement
    alpha_factor = decayed_factor * (1 + 0.2 * sign_consistency)
    
    return alpha_factor
