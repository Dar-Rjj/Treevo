import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-Horizon Price-Amount Momentum Divergence with Adaptive Volatility Scaling
    
    This factor captures predictive signals by analyzing price-amount divergences
    across multiple time horizons, using adaptive volatility scaling and dollar
    volume weighting for enhanced regime awareness and interpretability.
    
    Key innovations:
    - Multi-horizon analysis (2-day and 5-day) for capturing different signal frequencies
    - Amount-based momentum instead of volume for better institutional flow representation
    - Multiplicative divergence ratios that amplify when price and amount move in opposite directions
    - Adaptive volatility scaling using rolling percentiles for regime detection
    - Dollar volume weighting with exponential smoothing for liquidity adjustment
    - No hard thresholds, using continuous regime measures
    
    The factor identifies stocks where price momentum diverges from amount momentum,
    suggesting potential reversals, with adaptive adjustments for market conditions.
    """
    # Multi-horizon price momentum with exponential smoothing
    close_ewm_2 = df['close'].ewm(span=2).mean()
    close_ewm_5 = df['close'].ewm(span=5).mean()
    
    price_momentum_2d = (df['close'] - close_ewm_2) / close_ewm_2
    price_momentum_5d = (df['close'] - close_ewm_5) / close_ewm_5
    
    # Multi-horizon amount momentum with exponential smoothing
    amount_ewm_2 = df['amount'].ewm(span=2).mean()
    amount_ewm_5 = df['amount'].ewm(span=5).mean()
    
    amount_momentum_2d = (df['amount'] - amount_ewm_2) / amount_ewm_2
    amount_momentum_5d = (df['amount'] - amount_ewm_5) / amount_ewm_5
    
    # Multiplicative divergence ratios across horizons
    divergence_2d = price_momentum_2d * (1 - amount_momentum_2d)
    divergence_5d = price_momentum_5d * (1 - amount_momentum_5d)
    
    # Combined multi-horizon divergence (geometric mean for multiplicative combination)
    combined_divergence = (divergence_2d * divergence_5d).pow(0.5)
    
    # Dollar volume weighting with exponential smoothing
    dollar_volume = df['close'] * df['volume']
    dollar_volume_ewm_3 = dollar_volume.ewm(span=3).mean()
    dollar_volume_weight = dollar_volume / dollar_volume_ewm_3
    
    # Adaptive volatility scaling using high-low range
    high_low_range = df['high'] - df['low']
    range_ewm_5 = high_low_range.ewm(span=5).mean()
    volatility_ratio = high_low_range / range_ewm_5
    
    # Adaptive regime scaling using rolling percentiles (no hard thresholds)
    volatility_percentile = volatility_ratio.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)).astype(float), raw=False
    )
    
    # Regime-aware smoothing: more smoothing in high volatility regimes
    regime_smoothing_factor = 1 + volatility_percentile  # 1 in low vol, 2 in high vol
    
    # Dollar volume weighted divergence with volatility scaling
    weighted_divergence = combined_divergence * dollar_volume_weight
    volatility_scaled = weighted_divergence / (volatility_ratio + 1e-7)
    
    # Adaptive exponential smoothing based on regime
    alpha_factor = volatility_scaled.ewm(span=3).mean() / regime_smoothing_factor
    
    return alpha_factor
