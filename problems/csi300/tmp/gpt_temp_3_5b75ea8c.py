import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive Volatility-Weighted Momentum with Volume Acceleration
    # Novel factor blending momentum signals across multiple timeframes with volume-weighted volatility adjustment
    # Economic rationale: Stocks with accelerating volume and consistent momentum across timeframes tend to persist
    
    # Multi-timeframe momentum signals (3-day and 5-day)
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Volume acceleration (rate of change in volume momentum)
    vol_mom_3d = (df['volume'] - df['volume'].shift(3)) / (df['volume'].shift(3) + 1e-7)
    vol_mom_5d = (df['volume'] - df['volume'].shift(5)) / (df['volume'].shift(5) + 1e-7)
    vol_acceleration = vol_mom_3d - vol_mom_5d  # Positive when short-term volume growth exceeds long-term
    
    # Adaptive volatility using rolling percentiles (5-day window)
    returns = df['close'].pct_change()
    vol_5d = returns.rolling(window=5).std()
    vol_percentile = vol_5d.rolling(window=10).apply(lambda x: (x.rank(pct=True).iloc[-1]), raw=False)
    
    # Asymmetric momentum signals (stronger weight to upward moves)
    up_momentum = np.where(mom_3d > 0, mom_3d * 1.5, mom_3d)  # Amplify positive momentum
    down_momentum = np.where(mom_5d < 0, mom_5d * 0.7, mom_5d)  # Dampen negative momentum
    
    # Volume-weighted volatility normalization
    vol_weight = (df['volume'] / df['volume'].rolling(window=10).mean())  # Relative volume strength
    volatility_adjustment = vol_5d * (1 + vol_percentile)  # Higher weight to high volatility regimes
    
    # Multiplicative volume-weighting for momentum signals
    mom_vol_weighted = (up_momentum + down_momentum) * vol_weight
    
    # Range expansion signal (high-low momentum convergence)
    high_mom = (df['high'] - df['high'].shift(3)) / df['high'].shift(3)
    low_mom = (df['low'] - df['low'].shift(3)) / df['low'].shift(3)
    range_expansion = (high_mom - low_mom) * np.sign(mom_3d)  # Directional range expansion
    
    # Final alpha factor with signal convergence
    alpha = (
        mom_vol_weighted / (volatility_adjustment + 1e-7) +  # Volatility-normalized volume-weighted momentum
        vol_acceleration * np.sign(mom_3d) +  # Volume acceleration aligned with price direction
        range_expansion * 0.5  # Range expansion confirmation
    )
    
    return alpha
