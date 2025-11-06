import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Extract price and volume data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Calculate 5-day price momentum
    momentum_5d = (close.shift(1) - close.shift(5)) / close.shift(5)
    
    # Calculate 10-day average volatility (daily range normalized by close)
    daily_range = (high - low) / close
    avg_volatility_10d = daily_range.rolling(window=10, min_periods=1).mean()
    
    # Adjust momentum by volatility with epsilon to avoid division by zero
    epsilon = 1e-8
    volatility_adjusted_momentum = momentum_5d / (avg_volatility_10d + epsilon)
    
    # Calculate 5-day volume trend using linear regression
    def calc_volume_slope(vol_series):
        if len(vol_series) < 2:
            return 0
        x = np.arange(len(vol_series))
        slope, _, _, _, _ = linregress(x, vol_series)
        return np.sign(slope)
    
    volume_slope_sign = volume.rolling(window=5, min_periods=2).apply(
        calc_volume_slope, raw=False
    )
    
    # Combine volatility-adjusted momentum with volume confirmation
    factor = volatility_adjusted_momentum * volume_slope_sign.shift(1)
    
    return factor
