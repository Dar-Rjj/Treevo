import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Volatility-normalized momentum-volume alignment factor.
    Combines medium-term momentum with volume trend and volatility efficiency.
    Uses decay weights for robustness and penalizes price inefficiencies.
    """
    # Medium-term momentum (5-day for more stable signal)
    momentum = df['close'] / df['close'].shift(5) - 1
    
    # Volume trend alignment (5-day volume momentum)
    volume_trend = df['volume'] / df['volume'].shift(5) - 1
    
    # Volatility efficiency (how efficiently price moves relative to range)
    daily_volatility = (df['high'] - df['low']) / df['close']
    volatility_efficiency = abs(momentum) / (daily_volatility + 1e-7)
    
    # Decay weights for recent observations (exponential decay)
    decay_weights = 0.9 ** np.arange(5)
    
    # Apply decay to momentum and volume trend
    momentum_decayed = momentum.rolling(window=5).apply(
        lambda x: np.sum(x * decay_weights) / np.sum(decay_weights)
    )
    volume_trend_decayed = volume_trend.rolling(window=5).apply(
        lambda x: np.sum(x * decay_weights) / np.sum(decay_weights)
    )
    
    # Combine components with volatility normalization
    # Momentum × Volume alignment × Efficiency, normalized by volatility
    alpha_factor = (momentum_decayed * volume_trend_decayed * volatility_efficiency) / (daily_volatility + 1e-7)
    
    # Penalty for inefficiency (when momentum and volume trend diverge)
    alignment_penalty = np.sign(momentum_decayed) != np.sign(volume_trend_decayed)
    alpha_factor = alpha_factor * (1 - 0.3 * alignment_penalty)
    
    return alpha_factor
