import numpy as np
def heuristics_v2(df):
    # Calculate the ratio of close to open prices (Close/Open)
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Calculate the ratio of high to low prices (High/Low)
    df['high_low_ratio'] = df['high'] / df['low']
    
    # Calculate the exponential moving average (EMA) of trading volume over a period (e.g., 20 days)
    df['volume_ema'] = df['volume'].ewm(span=20, adjust=False).mean()
    
    # Calculate the difference between current day's volume and the EMA volume
    df['volume_diff'] = df['volume'] - df['volume_ema']
    
    # Compute the ratio of the current day's volume to the EMA volume
    df['volume_ratio'] = df['volume'] / df['volume_ema']
    
    # Compute the percentage change in close price over a short period (e.g., 5 days)
    df['close_pct_change_5d'] = df['close'].pct_change(periods=5)
    
    # Compute the percentage change in close price over a long period (e.g., 60 days)
    df['close_pct_change_60d'] = df['close'].pct_change(periods=60)
    
    # Determine the number of days the close price has been increasing or decreasing consecutively
    df['consecutive_days'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['consecutive_days'] = df['consecutive_days'].diff().ne(0).cumsum()
    df['consecutive_days'] = df.groupby('consecutive_days')['consecutive_days'].transform('size')
    
    # Calculate the rate of change (ROC) of the close price over a period (e.g., 14 days)
    df['roc_14d'] = df['close'].pct_change(periods=14)
    
    # Calculate the directional movement index (DMI) over a period (e.g., 14 days)
    df['high_diff'] = df['high'] - df['high'].shift(1)
    df['low_diff'] = df['low'].shift(1) - df['low']
    df['positive_dm'] = df['high_diff'].apply(lambda x: x if x > 0 else 0)
    df['negative_dm'] = df['low_diff'].apply(lambda x: x if x > 0 else 0)
    df['tr'] = np.maximum(np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['close'].shift(1))), np.abs(df['low'] - df['close'].shift(1)))
    df['smoothed_tr'] = df['tr'].rolling(window=14).sum()
    df['smoothed_positive_dm'] = df['positive_dm'].rolling(window=14).sum()
    df['smoothed_negative_dm'] = df['negative_dm'].rolling(window=14).sum()
    df['di_plus'] = (df['smoothed_positive_dm'] / df['smoothed_tr']) * 100
    df['di_minus'] = (df['smoothed_negative_dm'] / df['smoothed_tr']) * 100
    df['directional_movement_index'] = abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
    
    # Calculate the daily price range (High - Low)
    df['price_range'] = df['high'] - df['low']
    
    # Calculate the average true range over a period (e.g., 14 days)
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x[0], df['close'].shift(1)) - min(x[1], df['close'].shift(1)), axis=1)
    df['average_true_range'] = df['true_range'].rolling(window=14).mean()
    
    # Compute the standard deviation of returns over a period (e.g., 20 days)
