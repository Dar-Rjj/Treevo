import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Volatility-scaled momentum with volume-price divergence and regime detection
    Interpretation: Combines momentum adjusted for recent volatility, detects divergence 
    between price movement and volume patterns, and adapts to different market regimes 
    using short-term trend indicators. The factor identifies stocks with sustainable 
    momentum in appropriate volatility environments with confirming volume signals.
    """
    
    # Volatility-scaled momentum - 3-day return scaled by 5-day volatility
    momentum_3d = df['close'].pct_change(3)
    volatility_5d = df['close'].pct_change().rolling(window=5, min_periods=1).std()
    vol_scaled_momentum = momentum_3d / (volatility_5d + 1e-7)
    
    # Volume-price divergence - price change vs volume change correlation
    price_change = df['close'].pct_change()
    volume_change = df['volume'].pct_change()
    vol_price_divergence = price_change.rolling(window=3, min_periods=1).corr(volume_change)
    
    # Regime detection - short-term trend strength
    short_trend = df['close'].rolling(window=5, min_periods=1).apply(
        lambda x: (x[-1] - x[0]) / (np.std(x) + 1e-7) if len(x) > 1 else 0
    )
    
    # Adaptive weighting based on regime
    regime_weight = np.where(short_trend.abs() > 0.5, 1.2, 0.8)
    
    # Combine components with regime-adaptive weights
    alpha_factor = vol_scaled_momentum * vol_price_divergence * regime_weight
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
