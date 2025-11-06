import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Momentum Component
    close = df['close']
    momentum_5d = close / close.shift(5) - 1
    momentum_accel = momentum_5d - momentum_5d.shift(3)
    
    # Volatility Adjustment
    high = df['high']
    low = df['low']
    prev_close = close.shift(1)
    true_range = np.maximum(high, prev_close) - np.minimum(low, prev_close)
    volatility_10d = true_range.rolling(window=10, min_periods=10).std()
    vol_adjusted_momentum = momentum_5d / volatility_10d
    
    # Volume-Price Alignment
    volume = df['volume']
    
    def calc_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y_values = series.iloc[i-window+1:i+1].values
                x_values = np.arange(len(y_values))
                slope, _, _, _, _ = linregress(x_values, y_values)
                slopes.iloc[i] = slope
        return slopes
    
    volume_slope = calc_slope(volume, 5)
    price_slope = calc_slope(close, 5)
    alignment_strength = volume_slope * price_slope
    
    # Market Regime Classification
    atr_20 = true_range.rolling(window=20, min_periods=20).mean()
    atr_20_median = atr_20.rolling(window=60, min_periods=60).median()
    high_vol_regime = atr_20 > atr_20_median
    low_vol_regime = atr_20 <= atr_20_median
    
    # Factor Integration
    factor = pd.Series(index=df.index, dtype=float)
    factor[high_vol_regime] = vol_adjusted_momentum[high_vol_regime] * (1 + alignment_strength[high_vol_regime])
    factor[low_vol_regime] = vol_adjusted_momentum[low_vol_regime] * alignment_strength[low_vol_regime]
    
    return factor
