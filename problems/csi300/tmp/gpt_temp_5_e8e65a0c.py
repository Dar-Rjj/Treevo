import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum blend with volatility-scaled volume and regime detection
    # Interpretation: Combines short/medium momentum, volume adjusted for volatility, and market regime
    
    # Multi-timeframe momentum blend (5-day and 20-day)
    momentum_short = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_medium = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    momentum_blend = 0.6 * momentum_short + 0.4 * momentum_medium
    
    # Volatility-scaled volume (volume relative to recent volatility environment)
    returns_volatility = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    volume_avg = df['volume'].rolling(window=20, min_periods=10).mean()
    volatility_scaled_volume = (df['volume'] - volume_avg) / (returns_volatility + 1e-7)
    
    # Quantile-based regime detection (identify high/low volatility periods)
    volatility_regime = returns_volatility.rolling(window=60, min_periods=30).apply(
        lambda x: 1 if x.iloc[-1] > x.quantile(0.7) else (-1 if x.iloc[-1] < x.quantile(0.3) else 0)
    )
    
    # Clean multiplicative interaction with regime adjustment
    alpha_factor = momentum_blend * volatility_scaled_volume * (1 + 0.2 * volatility_regime)
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
