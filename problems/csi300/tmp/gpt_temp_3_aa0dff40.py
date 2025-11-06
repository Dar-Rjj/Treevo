import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced momentum-efficiency-volatility factor.
    Combines aligned short-term momentum, volume trends, and volatility with decay weights.
    Multiplicatively penalizes inefficiency while normalizing by volatility.
    """
    # Short-term momentum (3-day aligned momentum)
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    
    # Volume trend (3-day volume momentum)
    volume_trend = df['volume'] / df['volume'].shift(3) - 1
    
    # Volatility normalization (3-day rolling volatility)
    returns_1d = df['close'] / df['close'].shift(1) - 1
    volatility_3d = returns_1d.rolling(window=3).std()
    
    # Range efficiency with decay weights
    daily_range = (df['high'] - df['low']) / df['close']
    range_efficiency = abs(momentum_3d) / (daily_range + 1e-7)
    
    # Apply exponential decay to efficiency (recent efficiency more important)
    decay_weights = 0.9 ** np.arange(3)
    efficiency_decayed = range_efficiency.rolling(window=3).apply(
        lambda x: np.sum(x * decay_weights) / np.sum(decay_weights)
    )
    
    # Multiplicative combination with inefficiency penalty
    aligned_momentum = momentum_3d * np.sign(volume_trend)  # Align with volume trend
    volatility_normalized = aligned_momentum / (volatility_3d + 1e-7)
    
    # Final factor: volatility-normalized momentum × efficiency (penalize inefficiency)
    alpha_factor = volatility_normalized * efficiency_decayed
    
    return alpha_factor
