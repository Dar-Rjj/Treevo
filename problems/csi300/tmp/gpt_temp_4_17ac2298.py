import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Adaptive Momentum-Volume Divergence with Volatility-Scaled Regimes
    
    This factor enhances predictive power by combining momentum and volume signals
    with adaptive smoothing based on volatility regimes, while using dollar volume
    as a scaling mechanism for institutional flow sensitivity.
    
    Key improvements over v1:
    - Uses both positive and negative momentum-volume relationships
    - Adaptive smoothing based on volatility quintiles rather than fixed threshold
    - Combines price and volume momentum multiplicatively for stronger signals
    - Uses raw dollar volume scaling without normalization
    - Simpler regime detection using rolling volatility percentiles
    
    The factor identifies stocks where price momentum diverges from volume trends,
    with enhanced sensitivity during different market volatility conditions.
    """
    # 3-day price momentum (shorter horizon for responsiveness)
    price_momentum = df['close'].pct_change(periods=3)
    
    # 3-day volume momentum 
    volume_momentum = df['volume'].pct_change(periods=3)
    
    # Momentum-volume divergence (multiplicative combination)
    # Positive when price and volume move in opposite directions
    momentum_divergence = price_momentum * (1 - volume_momentum.abs())
    
    # Dollar volume scaling for institutional flow sensitivity
    dollar_volume = df['close'] * df['volume']
    
    # Volatility regime using 5-day rolling high-low range
    high_low_range = (df['high'] - df['low']) / df['close']
    rolling_volatility = high_low_range.rolling(window=5).mean()
    
    # Adaptive regime detection using volatility quintiles
    volatility_quantile = rolling_volatility.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Define volatility regimes
    high_vol_regime = volatility_quantile > 0.7
    medium_vol_regime = (volatility_quantile >= 0.3) & (volatility_quantile <= 0.7)
    low_vol_regime = volatility_quantile < 0.3
    
    # Regime-specific smoothing parameters
    # More smoothing in high volatility, less in low volatility
    span_high_vol = 4
    span_medium_vol = 3
    span_low_vol = 2
    
    # Apply dollar volume scaling to divergence signal
    volume_scaled_divergence = momentum_divergence * dollar_volume
    
    # Volatility normalization for cross-sectional comparability
    volatility_normalized = volume_scaled_divergence / (rolling_volatility + 1e-7)
    
    # Apply regime-aware exponential smoothing
    alpha_high_vol = volatility_normalized[high_vol_regime].ewm(span=span_high_vol).mean()
    alpha_medium_vol = volatility_normalized[medium_vol_regime].ewm(span=span_medium_vol).mean()
    alpha_low_vol = volatility_normalized[low_vol_regime].ewm(span=span_low_vol).mean()
    
    # Combine all regime factors
    alpha_factor = pd.concat([alpha_high_vol, alpha_medium_vol, alpha_low_vol]).sort_index()
    
    return alpha_factor
