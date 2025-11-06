import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced alpha factor combining extended timeframe momentum, sophisticated volume-price alignment,
    multi-period volatility normalization, and gap pattern analysis.
    
    This factor incorporates:
    - Multi-timeframe momentum (5-day and 15-day) for trend acceleration detection
    - Volume-price divergence/convergence with extended lookback periods
    - Adaptive volatility normalization using rolling percentiles
    - Gap persistence analysis measuring gap sustainability over multiple days
    
    Interpretable as: Stocks showing accelerating momentum supported by sustained volume-price
    alignment, adjusted for dynamic volatility conditions and gap persistence characteristics.
    """
    # Multi-timeframe momentum acceleration
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    momentum_15d = df['close'] / df['close'].shift(15) - 1
    momentum_acceleration = momentum_5d - momentum_15d
    
    # Volume-price alignment with extended confirmation
    price_trend_10d = df['close'] / df['close'].shift(10) - 1
    volume_trend_10d = df['volume'] / df['volume'].shift(10) - 1
    volume_price_alignment = np.sign(price_trend_10d) * volume_trend_10d
    
    # Adaptive volatility normalization using rolling percentile ranges
    daily_range = df['high'] - df['low']
    range_percentile = daily_range.rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.2)) / (x.quantile(0.8) - x.quantile(0.2) + 1e-7)
    )
    volatility_adjustment = 1 / (1 + range_percentile.abs())
    
    # Gap persistence analysis: overnight gap sustainability
    overnight_gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_persistence = overnight_gaps.rolling(window=3).apply(
        lambda x: x.iloc[0] if np.sign(x.iloc[0]) == np.sign(x.iloc[-1]) else 0
    )
    
    # Composite factor: momentum acceleration amplified by volume-price alignment,
    # adjusted for adaptive volatility and gap persistence
    alpha_factor = (momentum_acceleration * (1 + volume_price_alignment) * 
                   volatility_adjustment * (1 + gap_persistence))
    
    return alpha_factor
