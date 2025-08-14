def heuristics_v2(df):
    # Calculate the difference between closing price and opening price
    df['daily_diff'] = df['close'] - df['open']
    
    # 5-day moving average of the daily price difference
    df['5d_ma_daily_diff'] = df['daily_diff'].rolling(window=5).mean()
    
    # 20-day standard deviation of the 5-day moving average of the daily price difference
    df['20d_std_5d_ma_daily_diff'] = df['5d_ma_daily_diff'].rolling(window=20).std()
    
    # Volatility-adjusted trend strength
    df['vol_adj_trend_strength'] = df['5d_ma_daily_diff'] / df['20d_std_5d_ma_daily_diff']
    
    # 14-day Exponential Moving Average (EMA) of the closing prices
    df['14d_ema_close'] = df['close'].ewm(span=14, adjust=False).mean()
    
    # Difference between the current closing price and the 14-day EMA
    df['diff_close_14d_ema'] = df['close'] - df['14d_ema_close']
    
    # Count the number of days in the past month where the closing price was above the opening price
    df['up_days'] = (df['close'] > df['open']).astype(int)
    df['ratio_up_days'] = df['up_days'].rolling(window=20).sum() / 20
    
    # Calculate the highest high and lowest low over the last 20 days
    df['20d_highest_high'] = df['high'].rolling(window=20).max()
    df['20d_lowest_low'] = df['low'].rolling(window=20).min()
    
    # Range percentage
    df['range_percentage'] = (df['20d_highest_high'] - df['20d_lowest_low']) / df['20d_lowest_low']
    
    # Sum of volume on days when the closing price is higher than the opening price, divided by the total volume over the same period
    df['volume_up_days'] = df['volume'] * (df['close'] > df['open'])
    df['buying_pressure_ratio'] = df['volume_up_days'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Return based on the close-to-close price change
    df['daily_return'] = df['close'].pct_change()
