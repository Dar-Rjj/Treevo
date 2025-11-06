import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Momentum
    # Short-Term Momentum (5-day rate of change)
    short_momentum = data['close'].pct_change(periods=5)
    
    # Medium-Term Momentum (20-day rate of change)
    medium_momentum = data['close'].pct_change(periods=20)
    
    # Calculate Volatility Adjustment
    # Average Daily Range over 20 days
    daily_range = (data['high'] - data['low']) / data['close']
    rolling_volatility = daily_range.rolling(window=20, min_periods=10).mean()
    
    # Avoid division by zero
    volatility_adjusted = rolling_volatility.replace(0, np.nan)
    
    # Volatility-Adjusted Momentum
    short_vol_adj = short_momentum / volatility_adjusted
    medium_vol_adj = medium_momentum / volatility_adjusted
    
    # Calculate Momentum Divergence
    momentum_divergence = short_vol_adj - medium_vol_adj
    
    # Calculate Volume Trend (10-day linear regression slope)
    def volume_slope(volume_series):
        if len(volume_series) < 2:
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = linregress(x, volume_series)
        return slope
    
    volume_trend = data['volume'].rolling(window=10, min_periods=5).apply(
        volume_slope, raw=False
    )
    
    # Scale Divergence by Volume Trend
    factor = momentum_divergence * volume_trend
    
    return factor
