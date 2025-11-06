import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum with volatility-normalized regimes and adaptive volume thresholds.
    Uses quantile-based triggers for robust signals while avoiding data normalization.
    """
    
    # Multi-timeframe momentum (1-day and 3-day)
    mom_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Volatility regime detection using rolling standard deviation
    vol_5d = df['close'].pct_change().rolling(window=5).std()
    vol_regime = vol_5d.rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.quantile(0.7) else (-1 if x.iloc[-1] < x.quantile(0.3) else 0))
    
    # Volatility-normalized momentum (different normalization for different regimes)
    high_vol_threshold = vol_5d.rolling(window=20).quantile(0.8)
    low_vol_threshold = vol_5d.rolling(window=20).quantile(0.2)
    
    # Adaptive momentum scaling based on volatility regime
    vol_scaled_mom_1d = mom_1d / (vol_5d + 1e-7)
    vol_scaled_mom_3d = mom_3d / (vol_5d.rolling(window=3).mean() + 1e-7)
    
    # Volume regime using adaptive quantile thresholds
    vol_quantile_20d = df['volume'].rolling(window=20).quantile(0.7)
    vol_quantile_5d = df['volume'].rolling(window=5).quantile(0.7)
    
    # Volume momentum with regime awareness
    vol_mom_1d = (df['volume'] - df['volume'].shift(1)) / (df['volume'].shift(1) + 1e-7)
    high_volume_regime = (df['volume'] > vol_quantile_20d) | (df['volume'] > vol_quantile_5d)
    
    # Price-range efficiency (how efficiently price moves within daily range)
    daily_range = df['high'] - df['low']
    close_to_open = abs(df['close'] - df['open'])
    range_efficiency = close_to_open / (daily_range + 1e-7)
    
    # Quantile-based triggers
    mom_1d_quantile = mom_1d.rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.quantile(0.8) else (-1 if x.iloc[-1] < x.quantile(0.2) else 0))
    range_eff_quantile = range_efficiency.rolling(window=10).apply(lambda x: 1 if x.iloc[-1] > x.quantile(0.8) else 0)
    
    # Core factor construction
    # Blend short and medium-term momentum with volatility regime weighting
    momentum_blend = 0.6 * vol_scaled_mom_1d + 0.4 * vol_scaled_mom_3d
    
    # Apply volatility regime adjustments
    factor = momentum_blend.copy()
    
    # High volatility regime: reduce momentum weight, increase mean reversion
    high_vol_mask = vol_regime == 1
    factor = factor.mask(high_vol_mask, momentum_blend * 0.7 + range_efficiency * 0.3 * mom_1d_quantile)
    
    # Low volatility regime: pure momentum with volume confirmation
    low_vol_mask = vol_regime == -1
    factor = factor.mask(low_vol_mask, momentum_blend * (1 + 0.3 * high_volume_regime))
    
    # Apply quantile-based momentum triggers
    strong_up_mom = mom_1d_quantile == 1
    strong_down_mom = mom_1d_quantile == -1
    
    factor = factor.mask(strong_up_mom & high_volume_regime, factor * 1.2)
    factor = factor.mask(strong_down_mom & high_volume_regime, factor * 0.8)
    
    # Range efficiency enhancement
    high_efficiency = range_eff_quantile == 1
    factor = factor.mask(high_efficiency, factor * (1 + 0.15 * vol_regime))
    
    return factor
