import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced robust multi-timeframe momentum with regime-aware volume confirmation.
    
    Factor Logic:
    1. Multi-timeframe momentum (3d, 8d, 21d) using robust median-based calculations
    2. Volume acceleration confirmation across matching timeframes
    3. Volatility regime detection using robust range statistics
    4. Price-volume divergence detection
    5. Adaptive weighting based on market regime
    
    Interpretation:
    - Positive values: strong momentum with volume confirmation in appropriate regime
    - Negative values: momentum weakness, volume divergence, or regime mismatch
    - Robust statistics minimize outlier sensitivity
    - Regime-aware design adapts to different market conditions
    """
    
    # Robust multi-timeframe momentum using median-based price changes
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_8d = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    mom_21d = (df['close'] - df['close'].shift(21)) / df['close'].shift(21)
    
    # Volume acceleration using robust median comparisons
    vol_accel_3d = df['volume'] / df['volume'].rolling(window=5).median()
    vol_accel_8d = df['volume'] / df['volume'].rolling(window=10).median()
    vol_accel_21d = df['volume'] / df['volume'].rolling(window=21).median()
    
    # Volatility regime detection using robust range statistics
    daily_range = df['high'] - df['low']
    range_median = daily_range.rolling(window=21).median()
    range_mad = (daily_range - range_median).abs().rolling(window=21).median()
    vol_regime = (daily_range - range_median) / (range_mad + 1e-7)
    
    # Price-volume divergence detection
    price_trend_3d = df['close'].rolling(window=5).apply(lambda x: np.median(np.diff(x)))
    volume_trend_3d = df['volume'].rolling(window=5).apply(lambda x: np.median(np.diff(x)))
    pv_divergence_3d = np.sign(price_trend_3d) != np.sign(volume_trend_3d)
    
    price_trend_8d = df['close'].rolling(window=10).apply(lambda x: np.median(np.diff(x)))
    volume_trend_8d = df['volume'].rolling(window=10).apply(lambda x: np.median(np.diff(x)))
    pv_divergence_8d = np.sign(price_trend_8d) != np.sign(volume_trend_8d)
    
    # Regime classification
    high_vol_regime = vol_regime > 2.0
    low_vol_regime = vol_regime < -2.0
    normal_vol_regime = ~high_vol_regime & ~low_vol_regime
    
    # Regime-specific momentum weighting
    momentum_high_vol = 0.7 * mom_3d + 0.2 * mom_8d + 0.1 * mom_21d
    momentum_low_vol = 0.2 * mom_3d + 0.3 * mom_8d + 0.5 * mom_21d
    momentum_normal = 0.4 * mom_3d + 0.4 * mom_8d + 0.2 * mom_21d
    
    # Regime-specific volume weighting
    volume_high_vol = 0.7 * vol_accel_3d + 0.2 * vol_accel_8d + 0.1 * vol_accel_21d
    volume_low_vol = 0.2 * vol_accel_3d + 0.3 * vol_accel_8d + 0.5 * vol_accel_21d
    volume_normal = 0.4 * vol_accel_3d + 0.4 * vol_accel_8d + 0.2 * vol_accel_21d
    
    # Apply regime-based selection
    momentum_regime = pd.Series(index=df.index, dtype=float)
    volume_regime = pd.Series(index=df.index, dtype=float)
    
    momentum_regime[high_vol_regime] = momentum_high_vol[high_vol_regime]
    momentum_regime[low_vol_regime] = momentum_low_vol[low_vol_regime]
    momentum_regime[normal_vol_regime] = momentum_normal[normal_vol_regime]
    
    volume_regime[high_vol_regime] = volume_high_vol[high_vol_regime]
    volume_regime[low_vol_regime] = volume_low_vol[low_vol_regime]
    volume_regime[normal_vol_regime] = volume_normal[normal_vol_regime]
    
    # Base factor combining momentum and volume
    base_factor = momentum_regime * volume_regime
    
    # Apply divergence penalties
    divergence_penalty = pd.Series(1.0, index=df.index)
    divergence_penalty[pv_divergence_3d] = 0.8
    divergence_penalty[pv_divergence_8d] = 0.6
    
    # Final factor with divergence adjustment
    factor = base_factor * divergence_penalty
    
    return factor
