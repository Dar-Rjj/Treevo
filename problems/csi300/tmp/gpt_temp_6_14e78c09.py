import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Volatility-normalized momentum-volume synergy factor.
    Combines short-term momentum, volume confirmation, and volatility adjustment
    with decay weighting for robustness.
    """
    # 5-day window for short-term signals
    window = 5
    
    # Price momentum (5-day return)
    momentum = df['close'] / df['close'].shift(window) - 1
    
    # Volume momentum (5-day volume change)
    volume_momentum = df['volume'] / df['volume'].shift(window) - 1
    
    # Volatility measure (5-day average true range normalized by close)
    high_low_range = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low_range, high_prev_close, low_prev_close], axis=1).max(axis=1)
    volatility = true_range.rolling(window).mean() / df['close']
    
    # Price efficiency (momentum per unit volatility)
    price_efficiency = momentum / (volatility + 1e-7)
    
    # Volume confirmation (momentum supported by volume)
    volume_confirmation = momentum * volume_momentum
    
    # Decay weights (linear decay for recent observations)
    weights = np.linspace(1.0, 0.2, window)
    
    # Apply decay-weighted combination
    weighted_efficiency = price_efficiency.rolling(window).apply(
        lambda x: np.sum(x * weights) if not x.isna().any() else np.nan, raw=False
    )
    weighted_confirmation = volume_confirmation.rolling(window).apply(
        lambda x: np.sum(x * weights) if not x.isna().any() else np.nan, raw=False
    )
    
    # Multiplicative synergy factor
    alpha_factor = weighted_efficiency * weighted_confirmation
    
    return alpha_factor
