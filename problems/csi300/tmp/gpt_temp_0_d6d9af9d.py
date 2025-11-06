import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 3-day volatility-normalized momentum
    mom_3 = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    tr_3 = (df['high'] - df['low']).rolling(window=3).mean()
    vol_norm_mom_3 = mom_3 / (tr_3 + 1e-7)
    
    # 5-day volatility-normalized momentum
    mom_5 = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    tr_5 = (df['high'] - df['low']).rolling(window=5).mean()
    vol_norm_mom_5 = mom_5 / (tr_5 + 1e-7)
    
    # Blend 3-5 day volatility-normalized momentum
    momentum_blend = 0.6 * vol_norm_mom_3 + 0.4 * vol_norm_mom_5
    
    # 1-day volume acceleration (current vs previous day)
    vol_accel_1d = df['volume'] / df['volume'].shift(1)
    
    # Combine momentum blend with volume acceleration
    momentum_volume_combo = momentum_blend * vol_accel_1d
    
    # Weight by recent volatility (10-day rolling standard deviation)
    recent_volatility = df['close'].pct_change().rolling(window=10).std()
    volatility_weighted = momentum_volume_combo * recent_volatility
    
    # Detect extremes via 20-day price percentiles
    price_percentile_20d = df['close'].rolling(window=20).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    # Reversal signals based on price extremes
    oversold_reversal = price_percentile_20d < 0.2  # Bottom 20%
    overbought_reversal = price_percentile_20d > 0.8  # Top 20%
    
    # Apply reversal adjustments
    factor = volatility_weighted.mask(oversold_reversal, volatility_weighted * 1.8)
    factor = factor.mask(overbought_reversal, factor * 0.2)
    
    # Use robust z-scores (median-based) for final scaling
    rolling_median = factor.rolling(window=20, min_periods=10).median()
    rolling_mad = factor.rolling(window=20, min_periods=10).apply(
        lambda x: (x - x.median()).abs().median()
    )
    robust_zscore = (factor - rolling_median) / (rolling_mad + 1e-7)
    
    return robust_zscore
