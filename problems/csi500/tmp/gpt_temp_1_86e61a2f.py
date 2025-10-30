import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily high-low range
    daily_range = df['high'] - df['low']
    
    # Calculate daily open-close return
    open_close_return = (df['close'] - df['open']) / df['open']
    
    # Calculate intraday return skewness using 20-day rolling window
    def calculate_skewness(series):
        if len(series) < 3:
            return np.nan
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return 0
        skew = ((series - mean_val) ** 3).mean() / (std_val ** 3)
        return skew
    
    skewness = open_close_return.rolling(window=20, min_periods=3).apply(
        calculate_skewness, raw=False
    )
    
    # Calculate volume trend slope using linear regression over 10 days
    def volume_slope(volume_series):
        if len(volume_series) < 2:
            return np.nan
        x = np.arange(len(volume_series))
        slope = np.polyfit(x, volume_series, 1)[0]
        return slope
    
    volume_trend = df['volume'].rolling(window=10, min_periods=2).apply(
        volume_slope, raw=False
    )
    
    # Normalize volume trend to make it comparable with skewness
    volume_trend_normalized = volume_trend / df['volume'].rolling(window=10, min_periods=1).mean()
    
    # Combine skewness with volume trend
    # Positive volume trend amplifies positive skewness, negative amplifies negative skewness
    factor = skewness * (1 + volume_trend_normalized)
    
    return factor
