import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Compute Price Momentum
    close = df['close']
    
    # Short-Term Momentum (3-day)
    M_short = close - close.shift(3)
    
    # Long-Term Momentum (10-day)
    M_long = close - close.shift(10)
    
    # Apply Exponential Decay to Momentum
    D_short = np.exp(-3/10)
    D_long = np.exp(-10/10)
    M_decay = (M_short * D_short) - (M_long * D_long)
    
    # Compute Volume Acceleration
    volume = df['volume']
    
    # Volume Change (1-day)
    V_change = volume - volume.shift(1)
    
    # Volume Trend (5-day linear regression slope)
    def calc_volume_trend(vol_series):
        if len(vol_series) < 5 or vol_series.isna().any():
            return np.nan
        x = np.arange(5)
        slope, _, _, _, _ = stats.linregress(x, vol_series.values)
        return slope
    
    V_trend = volume.rolling(window=5).apply(calc_volume_trend, raw=False)
    
    # Identify Volatility Regime
    high = df['high']
    low = df['low']
    
    # True Range
    TR1 = high - low
    TR2 = abs(high - close.shift(1))
    TR3 = abs(low - close.shift(1))
    TR = pd.concat([TR1, TR2, TR3], axis=1).max(axis=1)
    
    # Volatility Regime Classification
    TR_median = TR.rolling(window=20).median()
    R = TR > TR_median  # True = High volatility, False = Low volatility
    
    # Generate Final Alpha Factor
    F_base = M_decay * V_change * V_trend
    
    # Apply Regime Adjustment
    alpha = F_base.copy()
    alpha[R] = F_base[R] * 1.5    # High volatility regime
    alpha[~R] = F_base[~R] * 0.7  # Low volatility regime
    
    return alpha
