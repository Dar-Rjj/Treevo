import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor: Adaptive Intraday Range Momentum with Volume-Price Divergence
    
    This factor combines multiple robust techniques:
    1. Intraday range momentum using high-low range relative to open price
    2. Exponential smoothing for noise reduction across all components
    3. Adaptive volatility regime detection using rolling percentiles
    4. Volume-price divergence detection with smoothed confirmation
    5. Multi-timeframe analysis for regime-adaptive signal strength
    
    Interpretation:
    - Positive values: Negative intraday range momentum with volume divergence in stable regimes
    - Negative values: Positive intraday range momentum with volume divergence in stable regimes
    - Magnitude reflects signal strength adjusted for volatility regime and volume confirmation
    
    Economic rationale:
    - Intraday range captures true daily price movement intensity better than close-only
    - Smoothing reduces noise while preserving meaningful trend information
    - Volatility regime adaptation prevents over-trading in high-volatility periods
    - Volume-price divergence identifies potential reversal points
    - Multi-timeframe analysis captures both short-term dynamics and medium-term context
    """
    
    # Smoothed intraday range momentum (5-period EMA of normalized daily range)
    intraday_range = (df['high'] - df['low']) / df['open']
    smoothed_range_momentum = intraday_range.ewm(span=5, adjust=False).mean()
    
    # Volume-price divergence with dual smoothing
    price_change = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    volume_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    raw_divergence = price_change - volume_change
    smoothed_divergence = raw_divergence.ewm(span=3, adjust=False).mean()
    
    # Adaptive volatility regime scaling using rolling percentile
    volatility_5d = intraday_range.rolling(window=5, min_periods=5).std()
    volatility_regime = volatility_5d.rolling(window=20, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.3)) / (x.quantile(0.7) - x.quantile(0.3) + 1e-7)
    )
    regime_scaling = 1 / (1 + abs(volatility_regime))
    
    # Multi-timeframe momentum confirmation
    short_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    medium_momentum = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    momentum_alignment = short_momentum.ewm(span=3, adjust=False).mean() * medium_momentum.ewm(span=5, adjust=False).mean()
    
    # Integrated factor with regime-adaptive weighting
    alpha_factor = -smoothed_range_momentum * smoothed_divergence * regime_scaling * momentum_alignment
    
    return alpha_factor
