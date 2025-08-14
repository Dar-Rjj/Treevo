import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the ratio of close to open prices (Close/Open)
    close_open_ratio = df['close'] / df['open']
    
    # Calculate the ratio of high to low prices (High/Low)
    high_low_ratio = df['high'] / df['low']
    
    # Calculate the average trading volume over a period (e.g., 20 days)
    avg_volume_20d = df['volume'].rolling(window=20).mean()
    
    # Calculate the difference between current day's volume and the average volume
    volume_diff = df['volume'] - avg_volume_20d
    
    # Compute the percentage change in close price over a short period (e.g., 5 days)
    pct_change_5d = df['close'].pct_change(periods=5)
    
    # Compute the percentage change in close price over a long period (e.g., 60 days)
    pct_change_60d = df['close'].pct_change(periods=60)
    
    # Determine the number of days the close price has been increasing or decreasing consecutively
    df['consecutive_trend'] = (df['close'].diff() > 0).astype(int).groupby((df['close'].diff() <= 0).cumsum()).cumsum()
    
    # Calculate the daily price range (High - Low)
    daily_range = df['high'] - df['low']
    
    # Calculate the average true range over a period (e.g., 14 days)
    true_range = df['high'].combine(df['close'].shift(1), max) - df['low'].combine(df['close'].shift(1), min)
    avg_true_range_14d = true_range.rolling(window=14).mean()
    
    # Compute the standard deviation of returns over a period (e.g., 20 days)
    returns = df['close'].pct_change()
    std_dev_20d = returns.rolling(window=20).std()
    
    # Calculate the on-balance volume (OBV) over a period
    obv = (df['close'] > df['close'].shift()).astype(int) * df['volume'] - (df['close'] < df['close'].shift()).astype(int) * df['volume']
    obv = obv.cumsum()
    
    # Determine the percentage change in OBV over a period
    pct_change_obv_20d = obv.pct_change(periods=20)
    
    # Identify if there is a gap between the previous day's close and the current day's open
    gap = df['open'] - df['close'].shift()
    
    # Determine the size and direction of the gap
    gap_size = gap.abs()
    gap_direction = gap.apply(lambda x: 'up' if x > 0 else ('down' if x < 0 else 'neutral'))
    
    # Combine all the sub-thoughts into a single alpha factor
    alpha_factor = (
        close_open_ratio +
        high_low_ratio +
        volume_diff +
        pct_change_5d +
        pct_change_60d +
        df['consecutive_trend'] +
        daily_range +
        avg_true_range_14d +
        std_dev_20d +
        obv +
        pct_change_obv_20d +
        gap_size +
        gap_direction.map({'up': 1, 'down': -1, 'neutral': 0})
    )
    
    return alpha_factor
