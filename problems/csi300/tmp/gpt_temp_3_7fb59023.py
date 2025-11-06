import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-Horizon Volatility-Weighted Price-Volume Divergence with Adaptive Regime Scaling
    
    This factor enhances predictive power by combining divergence signals across multiple
    time horizons while maintaining regime awareness and dollar volume sensitivity.
    Key innovations:
    - Multi-horizon analysis (2-day and 5-day) for capturing different reversal patterns
    - Volatility-weighted divergence ratios for regime-adaptive scaling
    - Multiplicative combination of horizon-specific signals
    - Dollar volume exponential weighting for institutional flow responsiveness
    - Adaptive regime detection using rolling volatility percentiles
    
    The factor identifies stocks with unsupported price movements across multiple timeframes,
    weighted by liquidity and adjusted for current market volatility conditions.
    """
    # Multi-horizon price momentum with exponential smoothing
    close_ewm_2 = df['close'].ewm(span=2).mean()
    close_ewm_5 = df['close'].ewm(span=5).mean()
    
    price_momentum_2d = (df['close'] - close_ewm_2) / close_ewm_2
    price_momentum_5d = (df['close'] - close_ewm_5) / close_ewm_5
    
    # Multi-horizon volume momentum with exponential smoothing
    volume_ewm_2 = df['volume'].ewm(span=2).mean()
    volume_ewm_5 = df['volume'].ewm(span=5).mean()
    
    volume_momentum_2d = (df['volume'] - volume_ewm_2) / volume_ewm_2
    volume_momentum_5d = (df['volume'] - volume_ewm_5) / volume_ewm_5
    
    # Price-volume divergence across horizons (negative correlation signals)
    divergence_2d = price_momentum_2d * (1 - volume_momentum_2d)
    divergence_5d = price_momentum_5d * (1 - volume_momentum_5d)
    
    # Multiplicative combination of horizon divergences
    multi_horizon_divergence = divergence_2d * divergence_5d
    
    # Dollar volume exponential weighting for liquidity responsiveness
    dollar_volume = df['close'] * df['volume']
    dollar_volume_ewm_3 = dollar_volume.ewm(span=3).mean()
    dollar_volume_weight = dollar_volume / dollar_volume_ewm_3
    
    # Volatility regime using adaptive high-low range percentiles
    high_low_range = df['high'] - df['low']
    range_rolling_10 = high_low_range.rolling(window=10, min_periods=5)
    volatility_percentile = range_rolling_10.apply(lambda x: (x.iloc[-1] > x.quantile(0.7)).astype(int), raw=False)
    
    # Volatility-weighted scaling
    range_ewm_5 = high_low_range.ewm(span=5).mean()
    volatility_weight = high_low_range / range_ewm_5
    
    # Dollar volume weighted multi-horizon divergence
    weighted_divergence = multi_horizon_divergence * dollar_volume_weight
    
    # Volatility-weighted factor with regime adaptation
    volatility_weighted_factor = weighted_divergence / (volatility_weight + 1e-7)
    
    # Regime-aware exponential smoothing using volatility percentiles
    high_vol_mask = volatility_percentile == 1
    low_vol_mask = volatility_percentile == 0
    
    # Different smoothing parameters based on volatility regime
    span_high_vol = 4
    span_low_vol = 2
    
    # Apply regime-specific smoothing
    alpha_factor_high_vol = volatility_weighted_factor[high_vol_mask].ewm(span=span_high_vol).mean()
    alpha_factor_low_vol = volatility_weighted_factor[low_vol_mask].ewm(span=span_low_vol).mean()
    
    # Combine regime-aware factors
    alpha_factor = pd.concat([alpha_factor_high_vol, alpha_factor_low_vol]).sort_index()
    
    return alpha_factor
