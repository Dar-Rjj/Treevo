import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Parameters
    N = 20  # Lookback period for momentum
    M = 10  # Lookback period for volume slope
    
    # Calculate N-day stock return
    stock_return = df['close'] / df['close'].shift(N) - 1
    
    # Calculate N-day market return (using close as market proxy)
    market_return = df['close'].rolling(N).apply(
        lambda x: x[-1] / x[0] - 1 if len(x) == N and x[0] != 0 else np.nan,
        raw=True
    )
    
    # Calculate residual momentum
    residual_momentum = stock_return - market_return
    
    # Calculate M-day volume slope using linear regression
    def calc_volume_slope(volume_series):
        if len(volume_series) < M or np.all(volume_series == volume_series[0]):
            return np.nan
        x = np.arange(M)
        y = volume_series.values
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    volume_slope = df['volume'].rolling(M).apply(
        calc_volume_slope, raw=False
    )
    
    # Normalize volume slope by average volume to make it scale-invariant
    avg_volume = df['volume'].rolling(M).mean()
    normalized_volume_slope = volume_slope / avg_volume
    
    # Combine residual momentum with volume acceleration
    factor = residual_momentum * normalized_volume_slope
    
    return factor
