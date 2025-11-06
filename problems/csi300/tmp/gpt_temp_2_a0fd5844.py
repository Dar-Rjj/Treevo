import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 3-5 day volatility-normalized momentum blend
    # 3-day momentum normalized by 5-day true range volatility
    mom_3 = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    tr_5 = (df['high'] - df['low']).rolling(window=5).mean()
    vol_norm_mom_3 = mom_3 / (tr_5 + 1e-7)
    
    # 5-day momentum normalized by 8-day true range volatility
    mom_5 = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    tr_8 = (df['high'] - df['low']).rolling(window=8).mean()
    vol_norm_mom_5 = mom_5 / (tr_8 + 1e-7)
    
    # Blend 3-5 day volatility-normalized momentum
    momentum_blend = 0.6 * vol_norm_mom_3 + 0.4 * vol_norm_mom_5
    
    # 1-day volume acceleration (current vs previous day)
    vol_accel_1d = df['volume'] / df['volume'].shift(1)
    
    # Combine momentum blend with volume acceleration
    momentum_volume_combo = momentum_blend * vol_accel_1d
    
    # Recent volatility weighting (5-day rolling standard deviation)
    recent_vol = df['close'].pct_change().rolling(window=5).std()
    
    # Apply volatility weighting
    volatility_weighted = momentum_volume_combo / (recent_vol + 1e-7)
    
    # Detect extremes via 20-day price percentiles
    price_percentile_20d = df['close'].rolling(window=20).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    # Extreme price levels for reversals
    extreme_low = price_percentile_20d < 0.2
    extreme_high = price_percentile_20d > 0.8
    
    # Robust z-scores using median and MAD
    def robust_zscore(series, window=10):
        median = series.rolling(window=window).median()
        mad = series.rolling(window=window).apply(lambda x: np.median(np.abs(x - np.median(x))))
        return (series - median) / (mad + 1e-7)
    
    # Apply robust z-score to the volatility weighted factor
    factor_zscore = robust_zscore(volatility_weighted, window=10)
    
    # Apply reversal logic based on price extremes
    factor = factor_zscore.mask(extreme_low, factor_zscore * 1.5)  # Enhance signals at lows
    factor = factor.mask(extreme_high, factor_zscore * 0.7)        # Suppress signals at highs
    
    return factor
