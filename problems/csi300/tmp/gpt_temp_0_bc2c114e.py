import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum blend
    # Short-term (5-day) and medium-term (20-day) momentum with volatility normalization
    momentum_short = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_medium = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Volatility-scaled volume - volume relative to rolling volatility
    vol_20d = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    volume_scaled = df['volume'] / (vol_20d * df['close'] + 1e-7)
    volume_norm = volume_scaled / volume_scaled.rolling(window=20, min_periods=10).mean()
    
    # Quantile-based regime detection using price range
    range_ratio = (df['high'] - df['low']) / df['close']
    range_regime = range_ratio.rolling(window=20, min_periods=10).apply(
        lambda x: 1.0 if x.iloc[-1] > x.quantile(0.7) else 
                 -1.0 if x.iloc[-1] < x.quantile(0.3) else 0.0
    )
    
    # Multiplicative combination with regime weighting
    momentum_blend = 0.6 * momentum_short + 0.4 * momentum_medium
    alpha_factor = momentum_blend * volume_norm * (1 + 0.2 * range_regime)
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
