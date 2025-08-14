import numpy as np
def heuristics_v2(df):
    # Calculate the ratio of close to open prices (Close/Open)
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Calculate the ratio of high to low prices (High/Low)
    df['high_low_ratio'] = df['high'] / df['low']
    
    # Calculate the exponential moving average (EMA) of trading volume over a period (e.g., 20 days)
    df['volume_ema_20'] = df['volume'].ewm(span=20, adjust=False).mean()
    
    # Calculate the difference between current day's volume and the EMA volume
    df['volume_diff_ema'] = df['volume'] - df['volume_ema_20']
    
    # Compute the percentage change in close price over a short period (e.g., 5 days)
    df['close_pct_change_5'] = df['close'].pct_change(5)
    
    # Compute the percentage change in close price over a long period (e.g., 60 days)
    df['close_pct_change_60'] = df['close'].pct_change(60)
    
    # Determine the number of days the close price has been increasing or decreasing consecutively
    df['close_consecutive_days'] = np.sign(df['close'].diff()).groupby((df['close'].diff() > 0).cumsum()).cumcount() + 1
    
    # Calculate the rate of change (ROC) of the close price over a period (e.g., 14 days)
    df['roc_14'] = df['close'].pct_change(14)
    
    # Calculate the daily price range (High - Low)
    df['price_range'] = df['high'] - df['low']
    
    # Calculate the average true range over a period (e.g., 14 days)
    df['tr'] = df[['high', 'low', 'close']].apply(lambda x: np.max(x) - np.min(x), axis=1)
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    
    # Compute the standard deviation of returns over a period (e.g., 20 days)
