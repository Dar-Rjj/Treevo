import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Momentum-Volume Convergence Divergence factor
    Combines price momentum convergence with volume trend acceleration
    """
    close = df['close']
    volume = df['volume']
    
    # Price Momentum Component
    # Short-term momentum
    mom_3d = close / close.shift(3) - 1
    mom_6d = close / close.shift(6) - 1
    
    # Momentum convergence
    momentum_convergence = (mom_3d / mom_6d) - 1
    
    # Volume Trend Component
    # Volume slope calculation (9-day window)
    def calc_volume_slope(vol_series):
        if len(vol_series) < 2:
            return np.nan
        x = np.arange(len(vol_series))
        slope, _, _, _, _ = linregress(x, vol_series)
        return slope
    
    volume_slope = volume.rolling(window=10, min_periods=2).apply(
        calc_volume_slope, raw=False
    )
    
    # Volume acceleration (second derivative via linear regression on slopes)
    def calc_slope_slope(slope_series):
        if len(slope_series) < 2:
            return np.nan
        x = np.arange(len(slope_series))
        slope, _, _, _, _ = linregress(x, slope_series)
        return slope * 100
    
    volume_acceleration = volume_slope.rolling(window=5, min_periods=2).apply(
        calc_slope_slope, raw=False
    )
    
    # Signal Integration
    # Multiply momentum convergence by volume acceleration
    raw_signal = momentum_convergence * volume_acceleration
    
    # Apply rolling rank transformation (20-day window)
    def rolling_rank(series):
        return series.rank(pct=True)
    
    ranked_signal = raw_signal.rolling(window=20, min_periods=1).apply(
        lambda x: rolling_rank(pd.Series(x)).iloc[-1] if len(x) > 0 else 0.5, 
        raw=False
    )
    
    # Zero-center the signal
    final_signal = ranked_signal - 0.5
    
    return final_signal
