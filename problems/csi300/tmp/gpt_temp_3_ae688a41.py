import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Volatility-Normalized Momentum
    # Calculate 5-day price momentum
    momentum = (df['close'] / df['close'].shift(5)) - 1
    
    # Compute 5-day average daily range
    daily_range = df['high'] - df['low']
    avg_range = daily_range.rolling(window=5).mean()
    
    # Normalize momentum
    normalized_momentum = momentum / avg_range
    
    # Volume Confirmation Signal
    # Calculate 5-day volume slope using linear regression
    def calc_volume_slope(volume_series):
        if len(volume_series) < 5 or volume_series.isna().any():
            return np.nan
        x = np.arange(5)
        slope, _, _, _, _ = linregress(x, volume_series.values)
        return slope
    
    volume_slope = df['volume'].rolling(window=5).apply(calc_volume_slope, raw=False)
    
    # Generate confirmation flag
    volume_confirmation = np.where(
        (normalized_momentum > 0) & (volume_slope > 0), 1,
        np.where((normalized_momentum < 0) & (volume_slope < 0), 1,
                np.where((normalized_momentum > 0) & (volume_slope < 0), -1,
                        np.where((normalized_momentum < 0) & (volume_slope > 0), -1, 0)))
    )
    
    # Regime Detection
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate 20-day ATR and 60-day ATR median
    atr_20 = true_range.rolling(window=20).mean()
    atr_60_median = true_range.rolling(window=60).median()
    
    # Classify regime
    regime = np.where(atr_20 > atr_60_median, 'high_vol', 'low_vol')
    
    # Adaptive Factor Combination
    # Base factor
    base_factor = normalized_momentum * volume_confirmation
    
    # Apply regime-based weighting
    final_factor = np.where(
        regime == 'high_vol',
        0.7 * base_factor,
        1.3 * base_factor
    )
    
    return pd.Series(final_factor, index=df.index)
