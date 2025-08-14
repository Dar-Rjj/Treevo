import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate the difference between the close price of day t and the close price of day t-1
    df['close_diff'] = df['close'].diff()
    
    # Calculate the average of close prices over a certain lookback period (e.g., 5 days) and subtract it from the current close price
    df['close_avg_5d'] = df['close'].rolling(window=5).mean()
    df['close_to_avg_5d'] = df['close'] - df['close_avg_5d']
    
    # Calculate the slope of the linear regression of the close prices over a certain lookback period (e.g., 20 days)
    def calculate_slope(series, window=20):
        slopes = [linregress(range(window), series[i:i+window]).slope for i in range(len(series) - window + 1)]
        return pd.Series([np.nan] * (window - 1) + slopes, index=series.index)
    df['close_slope_20d'] = calculate_slope(df['close'])
    
    # Compute the high-low range as a measure of daily volatility
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate the ratio of the current close price to the 20-day moving average of the close price
    df['close_ma_20d'] = df['close'].rolling(window=20).mean()
    df['close_to_ma_20d_ratio'] = df['close'] / df['close_ma_20d']
    
    # Count the number of days in the past 10 days where the close price is above the open price
    df['close_above_open'] = df['close'] > df['open']
    df['up_days_count_10d'] = df['close_above_open'].rolling(window=10).sum()
    
    # Calculate the ratio of today's volume to the 20-day moving average of the volume
    df['volume_ma_20d'] = df['volume'].rolling(window=20).mean()
    df['volume_to_ma_20d_ratio'] = df['volume'] / df['volume_ma_20d']
    
    # Compute the difference between today's volume and yesterday's volume
    df['volume_diff'] = df['volume'].diff()
    
    # Determine the number of days in the last 10 days where the volume was higher than its 20-day moving average
    df['volume_above_ma_20d'] = df['volume'] > df['volume_ma_20d']
    df['high_volume_days_count_10d'] = df['volume_above_ma_20d'].rolling(window=10).sum()
    
    # Combine all factors into a single alpha factor
    alpha_factor = (
        df['close_diff'] +
        df['close_to_avg_5d'] +
        df['close_slope_20d'] +
        df['high_low_range'] +
        df['close_to_ma_20d_ratio'] +
        df['up_days_count_10d'] +
        df['volume_to_ma_20d_ratio'] +
        df['volume_diff'] +
        df['high_volume_days_count_10d']
    )
    
    return alpha_factor.dropna()
