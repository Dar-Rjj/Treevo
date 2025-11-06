import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-Horizon Price-Volume-Trend Divergence with Adaptive Regime Scaling
    
    This factor enhances predictive power by combining divergences across multiple time horizons
    with adaptive regime detection and dollar volume weighting.
    
    Key innovations:
    - Multi-horizon analysis (2-day and 5-day) for capturing different reversal patterns
    - Price-volume-trend triple divergence for stronger mean-reversion signals
    - Adaptive regime detection using rolling volatility percentiles
    - Multiplicative combination of horizon-specific divergences
    - Dollar volume exponential weighting for institutional flow sensitivity
    
    The factor identifies stocks with unsupported price movements across multiple timeframes,
    with enhanced regime awareness and liquidity responsiveness.
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
    
    # Multi-horizon price-volume divergence
    pv_divergence_2d = price_momentum_2d * (1 - volume_momentum_2d)
    pv_divergence_5d = price_momentum_5d * (1 - volume_momentum_5d)
    
    # Price trend component (close vs open momentum)
    open_ewm_2 = df['open'].ewm(span=2).mean()
    price_trend_2d = (df['close'] - open_ewm_2) / open_ewm_2
    
    # Triple divergence: price-volume-trend combination
    triple_divergence_2d = pv_divergence_2d * (1 - price_trend_2d)
    triple_divergence_5d = pv_divergence_5d * (1 - price_trend_2d)
    
    # Multi-horizon multiplicative combination
    horizon_combined = triple_divergence_2d * triple_divergence_5d
    
    # Dollar volume exponential weighting
    dollar_volume = df['close'] * df['volume']
    dollar_volume_ewm_3 = dollar_volume.ewm(span=3).mean()
    dollar_weight = dollar_volume / dollar_volume_ewm_3
    
    # Adaptive regime detection using rolling volatility percentiles
    high_low_range = df['high'] - df['low']
    range_rolling_10 = high_low_range.rolling(window=10, min_periods=5)
    volatility_percentile = range_rolling_10.apply(lambda x: (x.iloc[-1] > x.quantile(0.7)).astype(int))
    
    # Volatility scaling with regime adjustment
    range_ewm_3 = high_low_range.ewm(span=3).mean()
    volatility_scale = high_low_range / (range_ewm_3 + 1e-7)
    
    # Regime-dependent scaling factors
    high_vol_regime = volatility_percentile == 1
    regime_scale = high_vol_regime.astype(int) * 0.7 + (~high_vol_regime).astype(int) * 1.3
    
    # Apply dollar volume weighting and regime scaling
    weighted_divergence = horizon_combined * dollar_weight
    regime_adjusted = weighted_divergence * regime_scale
    
    # Volatility normalization with regime awareness
    volatility_normalized = regime_adjusted / (volatility_scale + 1e-7)
    
    # Adaptive smoothing based on regime
    alpha_factor_high_vol = volatility_normalized[high_vol_regime].ewm(span=4).mean()
    alpha_factor_low_vol = volatility_normalized[~high_vol_regime].ewm(span=2).mean()
    
    # Combine regime-aware factors
    alpha_factor = pd.concat([alpha_factor_high_vol, alpha_factor_low_vol]).sort_index()
    
    return alpha_factor
