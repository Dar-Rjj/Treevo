import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-Horizon Regime-Aware Price-Volume Momentum Divergence
    
    This factor enhances predictive power by combining multiple time horizons
    multiplicatively and using regime-dependent processing of price-volume relationships.
    Key innovations:
    - Multi-horizon signals (2-day and 5-day) combined multiplicatively
    - Price-volume momentum divergence instead of simple divergence
    - Adaptive volatility scaling using rolling percentiles
    - Dollar volume scaling for liquidity adjustment
    - Regime detection using volume volatility rather than price range
    
    The factor identifies stocks with strong price momentum but weakening volume support
    across multiple timeframes, suggesting potential trend exhaustion.
    """
    # Multi-horizon price momentum (2-day and 5-day)
    close_ewm_2 = df['close'].ewm(span=2).mean()
    close_ewm_5 = df['close'].ewm(span=5).mean()
    
    price_momentum_2d = (df['close'] - close_ewm_2) / close_ewm_2
    price_momentum_5d = (df['close'] - close_ewm_5) / close_ewm_5
    
    # Multi-horizon volume momentum (2-day and 5-day)
    volume_ewm_2 = df['volume'].ewm(span=2).mean()
    volume_ewm_5 = df['volume'].ewm(span=5).mean()
    
    volume_momentum_2d = (df['volume'] - volume_ewm_2) / volume_ewm_2
    volume_momentum_5d = (df['volume'] - volume_ewm_5) / volume_ewm_5
    
    # Price-volume momentum divergence (price up but volume down)
    pv_divergence_2d = price_momentum_2d * (1 - volume_momentum_2d)
    pv_divergence_5d = price_momentum_5d * (1 - volume_momentum_5d)
    
    # Multiplicative combination of horizons
    multi_horizon_divergence = pv_divergence_2d * pv_divergence_5d
    
    # Dollar volume scaling
    dollar_volume = df['close'] * df['volume']
    dollar_volume_ewm_5 = dollar_volume.ewm(span=5).mean()
    dollar_volume_scale = dollar_volume / dollar_volume_ewm_5
    
    # Volume volatility for regime detection (more stable than price range)
    volume_rolling_std = df['volume'].rolling(window=5).std()
    volume_mean = df['volume'].rolling(window=5).mean()
    volume_volatility = volume_rolling_std / (volume_mean + 1e-7)
    
    # Adaptive volatility scaling using rolling percentiles
    vol_vol_rolling_20 = volume_volatility.rolling(window=20)
    vol_vol_pct_80 = vol_vol_rolling_20.apply(lambda x: x.quantile(0.8))
    
    # High volatility regime when current volatility > 80th percentile
    high_vol_regime = volume_volatility > vol_vol_pct_80
    
    # Regime-dependent processing
    # In high volatility: use shorter smoothing and emphasize recent signals
    # In normal volatility: use balanced smoothing
    
    # Apply regime-aware exponential smoothing
    span_high_vol = 2  # More responsive in high volatility
    span_normal_vol = 3  # More smoothing in normal conditions
    
    # Dollar volume weighted multi-horizon divergence
    weighted_divergence = multi_horizon_divergence * dollar_volume_scale
    
    # Separate processing by regime
    high_vol_factor = weighted_divergence[high_vol_regime].ewm(span=span_high_vol).mean()
    normal_vol_factor = weighted_divergence[~high_vol_regime].ewm(span=span_normal_vol).mean()
    
    # Combine regime factors
    alpha_factor = pd.concat([high_vol_factor, normal_vol_factor]).sort_index()
    
    return alpha_factor
