import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum convergence with volume regime alignment and volatility-adaptive scaling
    # Economic intuition: Convergent momentum across timeframes, confirmed by volume regime persistence,
    # and adjusted for volatility conditions provides robust signals about trend quality and sustainability
    
    # Multi-timeframe momentum with geometrically spaced lookbacks
    momentum_2d = df['close'] / df['close'].shift(2) - 1
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    momentum_13d = df['close'] / df['close'].shift(13) - 1
    
    # Momentum convergence score: Measures alignment across timeframes using geometric mean of signed returns
    # Positive when all timeframes move in same direction with consistent magnitude scaling
    momentum_convergence = (
        np.sign(momentum_2d) * np.sign(momentum_5d) * np.sign(momentum_13d) *
        (abs(momentum_2d) * abs(momentum_5d) * abs(momentum_13d)) ** (1/3)
    )
    
    # Volume regime persistence using smooth regime transitions
    volume_ma_short = df['volume'].rolling(window=5).mean()
    volume_ma_medium = df['volume'].rolling(window=20).mean()
    volume_regime_ratio = volume_ma_short / (volume_ma_medium + 1e-7)
    
    # Smooth regime weighting using scaled hyperbolic tangent for bounded amplification
    # Maps volume regime to [0.5, 1.5] range with smooth transitions around normal regime
    regime_weight = 1.0 + 0.5 * np.tanh(volume_regime_ratio - 1)
    
    # Robust volatility scaling using median absolute deviation of true range
    true_range = np.maximum(
        np.maximum(df['high'] - df['low'], 
                  abs(df['high'] - df['close'].shift(1))),
        abs(df['low'] - df['close'].shift(1)))
    
    # Median-based volatility measure for outlier robustness
    volatility_mad = true_range.rolling(window=15).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    
    # Inverse volatility scaling with floor protection
    volatility_scale = 1 / (volatility_mad + 1e-7)
    
    # Combine convergent momentum with volume regime confirmation and robust volatility adaptation
    # Amplifies convergent signals during persistent volume regimes and low volatility conditions
    factor = momentum_convergence * regime_weight * volatility_scale
    
    return factor
