import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum factor with volatility normalization and adaptive volume thresholds.
    Dynamically detects market regimes and uses quantile-based triggers for robust signals.
    """
    
    # Multi-timeframe momentum (1, 3, 5 periods)
    mom_1 = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    mom_3 = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_5 = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Volatility normalization using rolling standard deviation
    vol_5d = df['close'].rolling(window=5).std()
    vol_10d = df['close'].rolling(window=10).std()
    
    # Normalized momentum signals
    norm_mom_1 = mom_1 / (vol_5d + 1e-7)
    norm_mom_3 = mom_3 / (vol_10d + 1e-7)
    norm_mom_5 = mom_5 / (vol_10d + 1e-7)
    
    # Adaptive volume thresholds using rolling quantiles
    vol_20d = df['volume'].rolling(window=20)
    vol_high_threshold = vol_20d.quantile(0.8)
    vol_low_threshold = vol_20d.quantile(0.2)
    
    # Volume regime detection
    high_vol_regime = df['volume'] > vol_high_threshold
    low_vol_regime = df['volume'] < vol_low_threshold
    normal_vol_regime = ~(high_vol_regime | low_vol_regime)
    
    # Momentum convergence/divergence detection
    momentum_aligned = ((mom_1 > 0) & (mom_3 > 0) & (mom_5 > 0)) | ((mom_1 < 0) & (mom_3 < 0) & (mom_5 < 0))
    momentum_divergence = ((mom_1 > 0) & (mom_5 < 0)) | ((mom_1 < 0) & (mom_5 > 0))
    
    # Price range efficiency (how much of daily range is utilized)
    daily_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (daily_range + 1e-7)
    
    # Quantile-based triggers for extreme price movements
    mom_1_quantile = mom_1.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    extreme_upside = mom_1_quantile > 0.9
    extreme_downside = mom_1_quantile < 0.1
    
    # Core factor construction with regime-based weighting
    base_factor = 0.4 * norm_mom_1 + 0.3 * norm_mom_3 + 0.3 * norm_mom_5
    
    # Regime adjustments
    factor = base_factor.copy()
    
    # High volume regime: emphasize short-term momentum
    factor = factor.mask(high_vol_regime, 0.6 * norm_mom_1 + 0.2 * norm_mom_3 + 0.2 * norm_mom_5)
    
    # Low volume regime: emphasize longer-term momentum
    factor = factor.mask(low_vol_regime, 0.2 * norm_mom_1 + 0.4 * norm_mom_3 + 0.4 * norm_mom_5)
    
    # Momentum alignment amplification
    factor = factor.mask(momentum_aligned, factor * 1.3)
    
    # Momentum divergence penalty
    factor = factor.mask(momentum_divergence, factor * 0.7)
    
    # Extreme price movement triggers
    factor = factor.mask(extreme_upside & (close_position > 0.7), factor * 1.5)
    factor = factor.mask(extreme_downside & (close_position < 0.3), factor * 1.5)
    
    # Range-bound mean reversion signals
    factor = factor.mask((close_position > 0.8) & momentum_aligned, factor * 0.8)
    factor = factor.mask((close_position < 0.2) & momentum_aligned, factor * 0.8)
    
    return factor
