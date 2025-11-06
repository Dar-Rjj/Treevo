import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Advanced momentum-volume-volatility interaction factor with regime detection.
    Combines 20-day momentum with 10-day volume acceleration and dynamic volatility filtering.
    Captures stocks with strong momentum supported by accelerating volume trends
    while filtering out high-volatility periods for more stable signals.
    """
    # 20-day momentum for robust trend persistence
    momentum = df['close'] / df['close'].shift(20) - 1
    
    # Volume acceleration: 5-day vs 10-day volume ratio with momentum alignment
    volume_5d = df['volume'].rolling(window=5).mean()
    volume_10d = df['volume'].rolling(window=10).mean()
    volume_acceleration = volume_5d / volume_10d - 1
    
    # Dynamic volatility regime using rolling percentile
    daily_range = (df['high'] - df['low']) / df['close']
    volatility_20d = daily_range.rolling(window=20).mean()
    volatility_rank = volatility_20d.rolling(window=60).apply(lambda x: (x[-1] > x.quantile(0.7)).astype(float), raw=False)
    
    # Volatility filter: penalize high volatility regimes
    volatility_filter = 1 - volatility_rank
    
    # Core interaction: momentum amplified by volume acceleration, filtered by volatility regime
    factor = momentum * volume_acceleration * volatility_filter
    
    return factor
