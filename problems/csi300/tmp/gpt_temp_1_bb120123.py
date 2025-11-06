import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-Horizon Regime-Adaptive Momentum with Volume Confirmation
    
    This factor combines multiple time horizons multiplicatively while adapting to
    volatility regimes and weighting by dollar volume. The factor captures stocks
    with consistent momentum signals across different timeframes, confirmed by
    volume patterns, while adjusting for market conditions.
    
    Key innovations:
    - Multiplicative combination of 3 horizons (ultra-short, short, medium)
    - Regime-adaptive volatility scaling using rolling percentiles
    - Dollar volume weighting for liquidity adjustment
    - Volume confirmation using amount-based signals
    - Simple, robust components with clear economic interpretation
    """
    # Ultra-short horizon (1-period momentum)
    mom_ultra = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Short horizon (3-period momentum)
    mom_short = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Medium horizon (5-period momentum)
    mom_medium = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Multiplicative horizon combination
    horizon_combined = mom_ultra * mom_short * mom_medium
    
    # Volume confirmation using amount (dollar volume momentum)
    amount_momentum = (df['amount'] - df['amount'].shift(3)) / df['amount'].shift(3)
    volume_confirmation = horizon_combined * (1 + amount_momentum)
    
    # Regime-aware volatility scaling using high-low range
    high_low_range = (df['high'] - df['low']) / df['close']
    vol_regime = high_low_range.rolling(window=10, min_periods=5).apply(
        lambda x: 2.0 if x.iloc[-1] > x.quantile(0.7) else 1.0 if x.iloc[-1] > x.quantile(0.3) else 0.5
    )
    
    # Dollar volume weighting (current period)
    dollar_volume = df['close'] * df['volume']
    dollar_weight = dollar_volume / dollar_volume.rolling(window=10, min_periods=5).mean()
    
    # Final factor: Multi-horizon momentum × Volume confirmation × Regime scaling × Dollar weight
    alpha_factor = volume_confirmation * vol_regime * dollar_weight
    
    return alpha_factor
