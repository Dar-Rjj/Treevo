import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(data):
    """
    Generate alpha factor combining volatility-adjusted trend strength and volume-weighted price acceleration
    """
    # Volatility-Adjusted Recent Trend Strength
    def volatility_adjusted_trend(data, N=20):
        # Calculate price trend using linear regression slope
        close_prices = data['close']
        trend_slopes = close_prices.rolling(window=N).apply(
            lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == N else np.nan
        )
        
        # Calculate recent volatility as average daily range
        daily_ranges = (data['high'] - data['low']) / data['close']
        volatility = daily_ranges.rolling(window=N).mean()
        
        # Adjust trend by volatility
        adjusted_trend = trend_slopes / (volatility + 1e-8)
        return adjusted_trend
    
    # Volume-Weighted Price Acceleration
    def volume_weighted_acceleration(data, M=20, N=5):
        close_prices = data['close']
        volume = data['volume']
        
        # Calculate rate of change over different periods
        roc_M = close_prices.pct_change(periods=M)
        roc_N = close_prices.pct_change(periods=N)
        
        # Calculate acceleration (difference in ROC normalized by time)
        acceleration = (roc_N - roc_M) / (M - N)
        
        # Calculate volume percentile
        volume_percentile = volume.rolling(window=N).apply(
            lambda x: stats.percentileofscore(x, x[-1]) / 100 if len(x) == N else np.nan
        )
        
        # Weight acceleration by volume activity
        weighted_acceleration = acceleration * volume_percentile
        return weighted_acceleration
    
    # Combine factors
    trend_factor = volatility_adjusted_trend(data)
    acceleration_factor = volume_weighted_acceleration(data)
    
    # Normalize and combine
    trend_normalized = (trend_factor - trend_factor.rolling(50).mean()) / (trend_factor.rolling(50).std() + 1e-8)
    acceleration_normalized = (acceleration_factor - acceleration_factor.rolling(50).mean()) / (acceleration_factor.rolling(50).std() + 1e-8)
    
    # Final factor (equal weighted combination)
    final_factor = 0.5 * trend_normalized + 0.5 * acceleration_normalized
    
    return final_factor
