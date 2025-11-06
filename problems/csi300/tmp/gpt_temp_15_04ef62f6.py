import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-Timeframe Momentum Divergence with Volume-Price Confirmation
    Captures divergences across multiple timeframes, confirmed by volume-price relationships
    and adjusted for adaptive volatility normalization
    """
    # Multi-timeframe momentum calculations
    momentum_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Divergence signals across timeframes
    divergence_short = momentum_1d - momentum_3d
    divergence_medium = momentum_3d - momentum_5d
    divergence_cross = divergence_short - divergence_medium
    
    # Volume-price relationship confirmation
    volume_trend = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    price_range_ratio = (df['high'] - df['low']) / df['close']
    volume_efficiency = momentum_1d / (volume_trend + 1e-7)
    
    # Adaptive volatility normalization using rolling percentiles
    daily_volatility = (df['high'] - df['low']) / df['close']
    vol_normalizer = daily_volatility.rolling(window=10, min_periods=1).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    )
    
    # Core divergence factor with volume confirmation
    divergence_factor = (
        0.4 * divergence_short + 
        0.35 * divergence_medium + 
        0.25 * divergence_cross
    ) * volume_efficiency
    
    # Final factor with volatility adjustment
    factor = divergence_factor / (vol_normalizer + 1e-7)
    
    return factor
