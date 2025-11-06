import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Momentum persistence normalized by volatility
    returns_5 = close.pct_change(5)
    returns_10 = close.pct_change(10)
    momentum_persistence = returns_5.rolling(10).corr(returns_10)
    volatility_normalizer = (high - low).rolling(10).std() / close.rolling(10).mean()
    normalized_momentum = momentum_persistence / (volatility_normalizer + 1e-8)
    
    # Volume entropy scaled mean reversion
    volume_entropy = -volume.rolling(10).apply(lambda x: (x/x.sum() * np.log(x/x.sum() + 1e-8)).sum())
    price_from_ma = (close - close.rolling(10).mean()) / close.rolling(10).std()
    mean_reversion = -price_from_ma * volume_entropy
    
    # Trend regime conditioning using price acceleration
    price_acceleration = close.pct_change(5) - close.pct_change(10)
    regime_weight = np.arctan(price_acceleration * 10)
    
    # Composite factor with regime adjustment
    heuristics_matrix = normalized_momentum * mean_reversion * regime_weight
    
    return heuristics_matrix
