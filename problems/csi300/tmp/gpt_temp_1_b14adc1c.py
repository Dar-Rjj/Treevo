import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 3-5 day volatility-normalized momentum blend
    mom_3 = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_5 = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Volatility estimation using true range
    tr_3 = (df['high'] - df['low']).rolling(window=3).mean()
    tr_5 = (df['high'] - df['low']).rolling(window=5).mean()
    
    # Volatility-normalized momentum
    vol_norm_mom_3 = mom_3 / (tr_3 + 1e-7)
    vol_norm_mom_5 = mom_5 / (tr_5 + 1e-7)
    
    # Blend 3-5 day momentum
    momentum_blend = 0.6 * vol_norm_mom_3 + 0.4 * vol_norm_mom_5
    
    # 1-day volume acceleration
    vol_accel = df['volume'] / df['volume'].shift(1)
    
    # Recent volatility weighting (5-day rolling volatility)
    recent_vol = (df['high'] - df['low']).rolling(window=5).std()
    vol_weight = 1 / (recent_vol + 1e-7)
    
    # Combine momentum with volume acceleration and volatility weight
    core_factor = momentum_blend * vol_accel * vol_weight
    
    # Extreme detection using 20-day price percentiles
    price_percentile_20 = df['close'].rolling(window=20).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    # Robust z-scores for momentum (using median and MAD)
    momentum_median = momentum_blend.rolling(window=20).median()
    momentum_mad = momentum_blend.rolling(window=20).apply(
        lambda x: (x - x.median()).abs().median()
    )
    momentum_z = (momentum_blend - momentum_median) / (momentum_mad + 1e-7)
    
    # Reversal signals based on price extremes
    oversold_reversal = price_percentile_20 < 0.2
    overbought_reversal = price_percentile_20 > 0.8
    
    # Apply reversal adjustments with momentum confirmation
    factor = core_factor.copy()
    factor = factor.mask(oversold_reversal & (momentum_z < -2), factor * 1.8)  # Strong bullish reversal
    factor = factor.mask(oversold_reversal & (momentum_z >= -2), factor * 1.3)  # Moderate bullish reversal
    factor = factor.mask(overbought_reversal & (momentum_z > 2), factor * 0.2)  # Strong bearish reversal
    factor = factor.mask(overbought_reversal & (momentum_z <= 2), factor * 0.6)  # Moderate bearish reversal
    
    return factor
