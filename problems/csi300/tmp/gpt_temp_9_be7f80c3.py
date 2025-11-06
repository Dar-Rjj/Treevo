import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Robust multi-timeframe momentum with dynamic regime adaptation and volume confirmation.
    
    Factor Logic:
    1. Robust volatility estimation using median-based methods
    2. Multi-timeframe momentum (3, 8, 15 days) normalized by robust volatility
    3. Dynamic weighting based on volatility regime using robust statistics
    4. Volume acceleration confirmation with robust outlier detection
    5. Price-volume divergence using robust correlation measures
    
    Interpretation:
    - Positive values indicate strong momentum with volume confirmation
    - Negative values suggest momentum weakness or reversal signals
    - Robust statistics reduce sensitivity to outliers
    - Dynamic regime adaptation responds to changing market conditions
    """
    
    # Robust volatility estimation using median absolute deviation
    price_range = df['high'] - df['low']
    vol_short = price_range.rolling(window=3).apply(lambda x: np.median(np.abs(x - np.median(x))))
    vol_medium = price_range.rolling(window=8).apply(lambda x: np.median(np.abs(x - np.median(x))))
    vol_long = price_range.rolling(window=15).apply(lambda x: np.median(np.abs(x - np.median(x))))
    
    # Robust momentum across multiple timeframes
    mom_short = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_medium = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    mom_long = (df['close'] - df['close'].shift(15)) / df['close'].shift(15)
    
    # Volatility-normalized momentum using robust volatility
    norm_mom_short = mom_short / (vol_short + 1e-7)
    norm_mom_medium = mom_medium / (vol_medium + 1e-7)
    norm_mom_long = mom_long / (vol_long + 1e-7)
    
    # Robust volume acceleration using median-based measures
    vol_accel_short = df['volume'] / df['volume'].rolling(window=3).median()
    vol_accel_medium = df['volume'] / df['volume'].rolling(window=8).median()
    vol_accel_long = df['volume'] / df['volume'].rolling(window=15).median()
    
    # Volume extremes detection using robust statistics
    vol_median = df['volume'].rolling(window=20).median()
    vol_mad = (df['volume'] - vol_median).abs().rolling(window=20).median()
    vol_extreme = -abs((df['volume'] - vol_median) / (vol_mad + 1e-7))
    
    # Dynamic regime detection using robust volatility ratio
    recent_vol = price_range.rolling(window=5).median()
    historical_vol = price_range.rolling(window=20).median()
    vol_regime = recent_vol / (historical_vol + 1e-7)
    
    # Dynamic weighting based on volatility regime
    high_vol_regime = vol_regime > 1.2
    low_vol_regime = vol_regime < 0.8
    normal_regime = ~(high_vol_regime | low_vol_regime)
    
    # Momentum blend with dynamic weights
    momentum_blend = pd.Series(index=df.index, dtype=float)
    momentum_blend[high_vol_regime] = (0.6 * norm_mom_short + 0.3 * norm_mom_medium + 0.1 * norm_mom_long)[high_vol_regime]
    momentum_blend[low_vol_regime] = (0.2 * norm_mom_short + 0.5 * norm_mom_medium + 0.3 * norm_mom_long)[low_vol_regime]
    momentum_blend[normal_regime] = (0.4 * norm_mom_short + 0.4 * norm_mom_medium + 0.2 * norm_mom_long)[normal_regime]
    
    # Volume blend with dynamic weights
    volume_blend = pd.Series(index=df.index, dtype=float)
    volume_blend[high_vol_regime] = (0.7 * vol_accel_short + 0.2 * vol_accel_medium + 0.1 * vol_accel_long)[high_vol_regime]
    volume_blend[low_vol_regime] = (0.2 * vol_accel_short + 0.5 * vol_accel_medium + 0.3 * vol_accel_long)[low_vol_regime]
    volume_blend[normal_regime] = (0.4 * vol_accel_short + 0.4 * vol_accel_medium + 0.2 * vol_accel_long)[normal_regime]
    
    # Price-volume divergence using robust correlation
    price_change = df['close'].pct_change(3)
    volume_change = df['volume'].pct_change(3)
    price_volume_divergence = price_change * volume_change
    
    # Combined factor with robust components
    factor = (
        momentum_blend * volume_blend + 
        0.2 * vol_extreme + 
        0.15 * price_volume_divergence
    )
    
    return factor
