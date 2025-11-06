import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Robust momentum acceleration with dynamic regime switching and volatility scaling
    
    # Robust momentum using median-based returns (5-day and 10-day)
    close_median_5 = df['close'].rolling(window=5, min_periods=3).median()
    close_median_10 = df['close'].rolling(window=10, min_periods=5).median()
    
    momentum_5d = (df['close'] - close_median_5) / close_median_5
    momentum_10d = (df['close'] - close_median_10) / close_median_10
    
    # Momentum acceleration with regime-based smoothing
    momentum_accel = momentum_5d - momentum_10d
    momentum_accel_smooth = momentum_accel.rolling(window=3, min_periods=2).median()
    
    # Robust volatility using median absolute deviation of returns
    returns = df['close'].pct_change()
    vol_mad = returns.rolling(window=20, min_periods=10).apply(lambda x: np.median(np.abs(x - np.median(x))))
    
    # Dynamic regime detection using price position and volume characteristics
    high_median_20 = df['high'].rolling(window=20, min_periods=10).median()
    low_median_20 = df['low'].rolling(window=20, min_periods=10).median()
    price_position = (df['close'] - low_median_20) / (high_median_20 - low_median_20 + 1e-7)
    
    volume_median_short = df['volume'].rolling(window=5, min_periods=3).median()
    volume_median_medium = df['volume'].rolling(window=20, min_periods=10).median()
    volume_regime = volume_median_short / volume_median_medium
    
    # Regime classification with clear thresholds
    trend_regime = np.where(
        price_position > 0.7, 1.0,  # Strong uptrend
        np.where(price_position < 0.3, -1.0, 0.0)  # Strong downtrend, else neutral
    )
    
    volume_regime_factor = np.where(
        volume_regime > 1.5, 2.0,  # High volume regime
        np.where(volume_regime < 0.7, 0.5, 1.0)  # Low volume regime, else normal
    )
    
    # Volatility regime based on robust MAD
    vol_regime = np.where(
        vol_mad > vol_mad.rolling(window=50, min_periods=25).quantile(0.75), 0.7,  # High vol: reduce exposure
        np.where(vol_mad < vol_mad.rolling(window=50, min_periods=25).quantile(0.25), 1.3, 1.0)  # Low vol: increase exposure
    )
    
    # Range efficiency with robust true range
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': np.abs(df['high'] - df['close'].shift(1)),
        'lc': np.abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    
    true_range_median = true_range.rolling(window=10, min_periods=5).median()
    price_change = np.abs(df['close'] - df['close'].shift(1))
    range_efficiency = price_change / (true_range_median + 1e-7)
    
    # Dynamic component weighting based on regime interactions
    momentum_weight = np.where(trend_regime != 0, 1.5, 1.0)
    volume_weight = volume_regime_factor
    volatility_weight = vol_regime
    efficiency_weight = np.where(range_efficiency > 0.8, 1.2, 1.0)
    
    # Clear component combination with regime-aware weights
    alpha_factor = (
        momentum_weight * momentum_accel_smooth * 
        volume_weight * (df['volume'] / volume_median_medium) * 
        volatility_weight * range_efficiency * 
        efficiency_weight
    )
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
