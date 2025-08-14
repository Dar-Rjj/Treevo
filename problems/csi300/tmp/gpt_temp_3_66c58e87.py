import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df, n=10, m=30, p=5, q=10):
    """
    Generate a novel and interpretable alpha factor based on the provided DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns (open, high, low, close, amount, volume) and index (date).
    n (int): Number of days to look back for the price momentum factor.
    m (int): Number of days to calculate the average volume for the volume weighting.
    p (int): Number of days for the moving sum of open-close differences.
    q (int): Number of days for the moving average of daily ranges.
    
    Returns:
    pd.Series: A pandas Series indexed by (date) representing the factor values.
    """
    
    # Calculate the weighted difference between the close price of day t and the close price n days ago
    df['close_n_days_ago'] = df['close'].shift(n)
    df['volume_avg_m_days'] = df['volume'].rolling(window=m).mean()
    df['volume_weight'] = df['volume'] / df['volume_avg_m_days']
    df['weighted_price_momentum'] = (df['close'] - df['close_n_days_ago']) * df['volume_weight']
    
    # Calculate the ratio of volume on day t to the average volume over the past m days
    # Multiply the volume ratio by the daily return (close price of day t minus open price of day t)
    df['daily_return'] = df['close'] - df['open']
    df['volume_ratio'] = df['volume'] / df['volume_avg_m_days']
    df['volume_weighted_daily_return'] = df['daily_return'] * df['volume_ratio']
    
    # Measure the difference between the open and close prices for each day up to t
    df['open_close_diff'] = df['close'] - df['open']
    # Compute the moving sum of these differences over a specific window (e.g., p days)
    df['moving_sum_open_close_diff'] = df['open_close_diff'].rolling(window=p).sum()
    # Calculate the trend of the volume over the same window
    df['volume_trend'] = df['volume'].rolling(window=p).apply(lambda x: linregress(range(p), x)[0], raw=False)
    # Multiply the moving sum by the trend of the volume over the same window
    df['volume_trend_adjusted_open_close'] = df['moving_sum_open_close_diff'] * df['volume_trend']
    
    # Calculate the range (High - Low) for day t
    df['range'] = df['high'] - df['low']
    # Examine the moving average of these ranges over q days
    df['moving_avg_range'] = df['range'].rolling(window=q).mean()
    # Adjust the moving average of the range by the average volume over the same q days
    df['volume_avg_q_days'] = df['volume'].rolling(window=q).mean()
    df['volatility_adjusted_volume'] = df['moving_avg_range'] * df['volume_avg_q_days']
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['weighted_price_momentum'] + 
                          df['volume_weighted_daily_return'] + 
                          df['volume_trend_adjusted_open_close'] + 
                          df['volatility_adjusted_volume'])
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('path_to_your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
