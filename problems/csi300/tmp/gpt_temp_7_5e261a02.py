import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum acceleration with volume breakout confirmation
    # Uses robust rolling statistics and cleaner interactions
    
    # Price momentum acceleration (5-day vs 10-day momentum difference)
    short_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    medium_momentum = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    momentum_acceleration = short_momentum - medium_momentum
    
    # Volatility normalization using robust rolling IQR
    price_range = df['high'] - df['low']
    volatility = price_range.rolling(window=10).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    
    # Volume breakout detection using rolling percentiles
    volume_median = df['volume'].rolling(window=20).median()
    volume_75pct = df['volume'].rolling(window=20).apply(lambda x: np.percentile(x, 75))
    volume_breakout = (df['volume'] > volume_75pct).astype(float) * ((df['volume'] - volume_median) / (volume_75pct - volume_median + 1e-7))
    
    # Dynamic weighting based on recent momentum persistence
    momentum_persistence = (df['close'].shift(1) > df['close'].shift(3)).rolling(window=5).sum() / 5
    
    # Combine components with cleaner interactions
    # Risk-adjusted momentum acceleration weighted by volume breakout strength
    risk_adjusted_acceleration = momentum_acceleration / (volatility + 1e-7)
    alpha_factor = risk_adjusted_acceleration * volume_breakout * (1 + momentum_persistence)
    
    return alpha_factor
