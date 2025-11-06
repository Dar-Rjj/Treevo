import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-Timeframe Momentum Persistence with Volume Confirmation and Volatility Normalization
    Combines momentum persistence signals across different time horizons with volume-based confirmation
    and volatility-adjusted returns to identify robust momentum continuation opportunities
    """
    
    # 1. Multi-timeframe momentum persistence ratios
    # Short-term vs medium-term momentum persistence (3-day vs 10-day)
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    momentum_10d = df['close'] / df['close'].shift(10) - 1
    momentum_persistence_ratio = momentum_3d / (momentum_10d + 1e-7)
    
    # 2. Volume confirmation across multiple timeframes
    # Current volume strength relative to short and medium-term averages
    volume_strength_short = df['volume'] / df['volume'].rolling(window=5).mean()
    volume_strength_medium = df['volume'] / df['volume'].rolling(window=15).mean()
    volume_confirmation_strength = volume_strength_short * volume_strength_medium
    
    # 3. Volatility-normalized momentum signals
    # Daily range-based volatility normalization
    daily_volatility = (df['high'] - df['low']) / df['close']
    normalized_momentum_3d = momentum_3d / (daily_volatility + 1e-7)
    normalized_momentum_10d = momentum_10d / (daily_volatility + 1e-7)
    
    # 4. Momentum acceleration with volatility adjustment
    # Rate of change in momentum adjusted for current volatility environment
    momentum_acceleration = (momentum_3d - momentum_10d) / (daily_volatility + 1e-7)
    
    # 5. Volume persistence indicator
    # Sustained volume strength over recent period
    volume_trend = df['volume'].rolling(window=5).apply(
        lambda x: np.sum(x.diff().fillna(0) > 0) / len(x)
    )
    
    # Combine factors: Strong momentum persistence confirmed by volume across timeframes,
    # normalized by volatility, with acceleration signals and volume trend confirmation
    alpha_factor = (
        momentum_persistence_ratio *
        volume_confirmation_strength *
        normalized_momentum_3d *
        momentum_acceleration *
        volume_trend
    )
    
    return alpha_factor
