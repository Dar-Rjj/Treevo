import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor blending multi-timeframe momentum with volume persistence,
    using regime-aware complementary signals and recent data weighting.
    
    This factor captures stocks with consistent momentum across multiple timeframes
    (short, medium, long) that are supported by persistent volume trends, while
    using volatility-adjusted signals for regime awareness.
    """
    # Multi-timeframe momentum components
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    momentum_21d = (df['close'] - df['close'].shift(21)) / df['close'].shift(21)
    
    # Volume persistence: 5-day vs 10-day volume momentum
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_10d_avg = df['volume'].rolling(window=10).mean()
    volume_persistence = (volume_5d_avg - volume_10d_avg) / (volume_10d_avg + 1e-7)
    
    # Regime-aware volatility scaling using rolling percentiles
    returns = df['close'].pct_change()
    vol_21d = returns.rolling(window=21).std()
    vol_regime = vol_21d.rolling(window=63).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
    
    # Recent data weighting using exponential smoothing
    momentum_weighted = (
        0.5 * momentum_5d.ewm(span=5).mean() + 
        0.3 * momentum_10d.ewm(span=10).mean() + 
        0.2 * momentum_21d.ewm(span=21).mean()
    )
    
    # Complementary signals: Volume persistence confirms momentum
    volume_confirmation = volume_persistence.ewm(span=5).mean()
    
    # Regime-adjusted factor construction
    # Higher weights in stable volatility regimes (negative vol_regime)
    regime_weight = 1.0 / (1.0 + np.exp(vol_regime))
    
    alpha_factor = momentum_weighted * volume_confirmation * regime_weight
    
    return alpha_factor
