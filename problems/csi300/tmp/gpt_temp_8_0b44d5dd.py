import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum with volume trend confirmation and adaptive volatility scaling
    # Economic intuition: Stocks exhibiting momentum across multiple time horizons,
    # supported by consistent volume trends, and adjusted for regime-dependent volatility
    # tend to show more reliable return persistence
    
    # Short-term momentum (3-day) for recent price acceleration
    momentum_short = df['close'] / df['close'].shift(3) - 1
    
    # Medium-term momentum (10-day) for established trend strength
    momentum_medium = df['close'] / df['close'].shift(10) - 1
    
    # Volume trend: current volume relative to expanding window average
    # Captures whether volume is consistently above historical levels
    volume_trend = df['volume'] / df['volume'].expanding().mean()
    
    # Volume momentum: acceleration in trading activity
    volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    
    # Adaptive volatility scaling using rolling percentile ranks
    # More robust to volatility regimes than simple averages
    true_range = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1) / df['close']
    
    # Volatility regime adjustment using percentile ranking
    volatility_rank = true_range.rolling(window=20).apply(
        lambda x: (x.rank(pct=True).iloc[-1]), raw=False
    )
    
    # Core alpha: Combined momentum signals enhanced by volume dynamics,
    # scaled by volatility regime sensitivity
    alpha_factor = (momentum_short + momentum_medium) * volume_trend * volume_momentum / (volatility_rank + 0.5)
    
    return alpha_factor
