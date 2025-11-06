import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate residual momentum
    # Price momentum: 20-day return
    momentum_period = 20
    price_momentum = df['close'].pct_change(periods=momentum_period)
    
    # Market return (using close as proxy for market index)
    market_return = df['close'].pct_change(periods=momentum_period)
    
    # Residual momentum = stock return - market return
    residual_momentum = price_momentum - market_return
    
    # Calculate volume acceleration
    # Volume trend: 10-day linear regression slope of volume
    volume_trend_period = 10
    
    def volume_slope(volume_series):
        if len(volume_series) < volume_trend_period:
            return np.nan
        x = np.arange(len(volume_series))
        slope = np.polyfit(x, volume_series, 1)[0]
        return slope
    
    volume_trend = df['volume'].rolling(window=volume_trend_period, min_periods=volume_trend_period).apply(
        volume_slope, raw=False
    )
    
    # Normalize volume trend by recent average volume
    volume_ma = df['volume'].rolling(window=volume_trend_period, min_periods=volume_trend_period).mean()
    normalized_volume_trend = volume_trend / volume_ma
    
    # Combine residual momentum with volume acceleration
    factor = residual_momentum * normalized_volume_trend
    
    return factor
