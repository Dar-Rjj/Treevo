import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the difference between today's closing price and yesterday's closing price
    df['close_diff'] = df['close'].diff()
    
    # Determine the high-low range for the day
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate the ratio of today's high-low range to the average high-low range over the past 10 days
    df['avg_high_low_range_10'] = df['high_low_range'].rolling(window=10).mean()
    df['volatility_ratio'] = df['high_low_range'] / df['avg_high_low_range_10']
    
    # Calculate the percentage change in volume from yesterday to today
    df['volume_change'] = df['volume'].pct_change()
    
    # Calculate the average true range (ATR) over the past 14 days
    df['tr'] = df[['high', 'low', 'close']].shift(1).apply(lambda x: max(x['high'], x['close']) - min(x['low'], x['close']), axis=1)
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    df['atr_ratio'] = df['tr'] / df['atr_14']
    
    # Calculate the cumulative sum of daily returns over the past 5 days
    df['daily_return'] = df['close'].pct_change()
    df['cumulative_return_5'] = df['daily_return'].rolling(window=5).sum()
    
    # Calculate the number of days the stock has closed higher than its 20-day moving average
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['above_ma_20'] = (df['close'] > df['ma_20']).rolling(window=20).sum()
    
    # Calculate the ratio of today's close to the highest high over the past 20 days
    df['highest_high_20'] = df['high'].rolling(window=20).max()
    df['close_to_highest_high'] = df['close'] / df['highest_high_20']
    
    # Calculate the ratio of today's close to the lowest low over the past 20 days
    df['lowest_low_20'] = df['low'].rolling(window=20).min()
    df['close_to_lowest_low'] = df['close'] / df['lowest_low_20']
    
    # Calculate the RSI (Relative Strength Index) over the past 14 days
    df['rsi_14'] = df['daily_return'].rolling(window=14).apply(lambda x: 100 - (100 / (1 + (x[x > 0].sum() / -x[x < 0].sum())))
    
    # Calculate the difference between the highest high and the lowest low over the past 20 days
    df['trading_range'] = df['highest_high_20'] - df['lowest_low_20']
    
    # Combine all the factors into a single alpha factor
    df['alpha_factor'] = (
        df['close_diff'] * 0.1 +
        df['volatility_ratio'] * 0.1 +
        df['volume_change'] * 0.1 +
        df['atr_ratio'] * 0.1 +
        df['cumulative_return_5'] * 0.1 +
        df['above_ma_20'] * 0.1 +
        df['close_to_highest_high'] * 0.1 +
        df['close_to_lowest_low'] * 0.1 +
        df['rsi_14'] * 0.1 +
        df['trading_range'] * 0.1
    )
    
    return df['alpha_factor']
