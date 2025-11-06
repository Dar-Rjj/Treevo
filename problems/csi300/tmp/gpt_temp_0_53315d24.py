import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Adaptive multi-timeframe momentum with volatility-scaled volume synergy
    Interpretation: Combines short-term (1-day) and medium-term (3-day) momentum signals,
    weighted by volatility regime and volume efficiency. The factor adapts to market conditions
    by emphasizing shorter momentum in high volatility and longer momentum in low volatility,
    while using volume normalized by price range as confirmation.
    """
    
    # Multi-timeframe momentum signals
    momentum_short = df['close'].pct_change(1)
    momentum_medium = df['close'].pct_change(3)
    
    # Volatility regime detection using 5-day rolling standard deviation
    volatility = df['close'].pct_change().rolling(window=5, min_periods=3).std()
    volatility_rank = volatility.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Adaptive weighting based on volatility regime
    # Higher weight to short momentum in high volatility, medium momentum in low volatility
    short_weight = volatility_rank
    medium_weight = 1 - volatility_rank
    
    # Volume efficiency - volume normalized by price range and average amount
    daily_range = df['high'] - df['low']
    avg_price = (df['high'] + df['low'] + df['close']) / 3
    volume_efficiency = df['volume'] / (daily_range * avg_price + 1e-7)
    
    # Volatility scaling for volume component
    volume_scaled = volume_efficiency / (volatility + 1e-7)
    
    # Blend multi-timeframe signals with regime-aware weights
    blended_momentum = (momentum_short * short_weight + momentum_medium * medium_weight)
    
    # Combine with volatility-scaled volume synergy
    alpha_factor = blended_momentum * volume_scaled
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
