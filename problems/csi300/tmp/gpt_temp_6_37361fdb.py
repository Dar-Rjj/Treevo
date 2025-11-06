import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-normalized momentum with volume divergence and regime detection
    # Economic intuition: Momentum signals are more reliable when volatility-adjusted,
    # confirmed by unusual volume activity, and during stable market regimes
    
    # Volatility-normalized momentum - 10-day return scaled by 20-day volatility
    returns_10d = df['close'].pct_change(periods=10)
    volatility_20d = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    momentum_norm = returns_10d / (volatility_20d + 1e-7)
    
    # Volume divergence - current volume percentile vs 20-day volume distribution
    volume_percentile = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: (x[-1] - x.mean()) / (x.std() + 1e-7) if x.std() > 0 else 0
    )
    
    # Regime shift detection - volatility regime change indicator
    short_vol = df['close'].pct_change().rolling(window=5, min_periods=3).std()
    long_vol = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    regime_stability = (long_vol - short_vol) / (long_vol + 1e-7)  # Positive when stable regime
    
    # Multiplicative combination for robustness
    alpha_factor = momentum_norm * volume_percentile * regime_stability
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
