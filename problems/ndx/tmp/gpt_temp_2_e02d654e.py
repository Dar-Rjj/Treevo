import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Simple Moving Average (SMA) of closing prices over a 20-day period
    df['SMA_20_close'] = df['close'].rolling(window=20).mean()
    
    # Difference between the most recent closing price and the 20-day SMA
    df['price_momentum'] = df['close'] - df['SMA_20_close']
    
    # Calculate 20-day moving average of volume
    df['SMA_20_volume'] = df['volume'].rolling(window=20).mean()
    
    # Ratio of the current day's volume to the 20-day volume average
    df['volume_ratio'] = df['volume'] / df['SMA_20_volume']
    
    # Compute the True Range for each day
    df['true_range'] = df[['high', 'low']].diff(axis=1).iloc[:, 0].abs() + \
                      df[['high', 'close']].diff(axis=1).iloc[:, 0].abs() + \
                      df[['low', 'close']].diff(axis=1).iloc[:, 0].abs()
    df['true_range'] = df['true_range'].max(axis=1)
    
    # Sum of the last 20 days' True Ranges as a measure of volatility
    df['volatility_cluster'] = df['true_range'].rolling(window=20).sum()
    
    # Determine the difference between the highest high and lowest low over the past 20 days
    df['high_low_diff_20'] = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
    
    # Compare today’s close with the 20-day high and 20-day low to gauge market sentiment
    df['20_day_high'] = df['high'].rolling(window=20).max()
    df['20_day_low'] = df['low'].rolling(window=20).min()
    df['sentiment'] = (df['close'] - df['20_day_low']) / (df['20_day_high'] - df['20_day_low'])
    
    # Use (High - Low) / (Close - Open) on a daily basis to identify potential bullish or bearish patterns
    df['daily_pattern'] = (df['high'] - df['low']) / (df['close'] - df['open'])
    
    # Calculate total amount traded in the last 20 days
    df['cumulative_amount_20'] = df['amount'].rolling(window=20).sum()
    
    # Daily change in cumulative amount traded to detect unusual trading activity
    df['daily_change_cumulative_amount'] = df['cumulative_amount_20'].diff()
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['price_momentum'] + 
                          df['volume_ratio'] + 
                          df['volatility_cluster'] + 
                          df['sentiment'] + 
                          df['daily_pattern'] + 
                          df['daily_change_cumulative_amount'])
    
    return df['alpha_factor']
