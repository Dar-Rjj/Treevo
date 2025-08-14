import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the 20-day moving average of close prices
    df['close_20ma'] = df['close'].rolling(window=20).mean()
    
    # Compute the percentage change from today's close to the 20-day moving average
    df['close_20ma_pct_change'] = (df['close'] - df['close_20ma']) / df['close_20ma']
    
    # Calculate the ratio of today's volume to the 20-day average volume
    df['volume_20ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_20ma']
    
    # Define thresholds for percentage deviation and volume ratio
    pct_change_threshold = 0.05
    volume_ratio_threshold = 1.5
    
    # Introduce a condition based on the 20-day moving average trend
    df['reversal_signal'] = ((df['close_20ma_pct_change'] > pct_change_threshold) & 
                             (df['volume_ratio'] > volume_ratio_threshold) & 
                             ((df['close_20ma'].diff() > 0) & (df['close'] < df['close_20ma']) |
                              (df['close_20ma'].diff() < 0) & (df['close'] > df['close_20ma'])))
    df['reversal_signal'] = df['reversal_signal'].astype(int)
    
    # Calculate the daily performance indicator
    df['daily_performance'] = (df['close'] - df['open']) / df['open']
    
    # Accumulate these daily returns over a rolling window of 10 days
    df['cumulative_return_10d'] = df['daily_performance'].rolling(window=10).sum()
    
    # Check if the cumulative return over the past 10 days switches sign
    df['sentiment_shift'] = df['cumulative_return_10d'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    
    # Calculate the range as the difference between the high and the low of the day
    df['range'] = df['high'] - df['low']
    
    # Calculate the 20-day average range
    df['range_20ma'] = df['range'].rolling(window=20).mean()
    
    # Develop a ratio of today's range to the 20-day average range
    df['range_ratio'] = df['range'] / df['range_20ma']
    
    # Calculate the true range
    df['true_range'] = df[['high' - 'low', 
                           'high' - df['close'].shift(1), 
                           df['close'].shift(1) - 'low']].max(axis=1)
    
    # Calculate the 20-day average true range
    df['true_range_20ma'] = df['true_range'].rolling(window=20).mean()
    
    # Calculate the ratio of today's range to the 20-day average true range
    df['range_true_ratio'] = df['range'] / df['true_range_20ma']
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['reversal_signal'] + 
                          df['sentiment_shift'] + 
                          (df['range_ratio'] - 1) * 10 + 
                          (df['range_true_ratio'] - 1) * 10)
    
    return df['alpha_factor']
