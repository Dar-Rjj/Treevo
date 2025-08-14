def heuristics_v2(df):
    # Calculate the difference between closing price and opening price
    df['daily_price_diff'] = df['close'] - df['open']
    
    # 5-day moving average of the daily price difference
    df['5day_ma_price_diff'] = df['daily_price_diff'].rolling(window=5).mean()
    
    # 20-day standard deviation of the 5-day moving average of the daily price difference
    df['20day_std_5day_ma_price_diff'] = df['5day_ma_price_diff'].rolling(window=20).std()
    
    # Volatility-Adjusted Trend Strength
    df['vol_adj_trend_strength'] = df['5day_ma_price_diff'] / df['20day_std_5day_ma_price_diff']
    
    # 14-day Exponential Moving Average (EMA) of the closing prices
    df['14day_ema_close'] = df['close'].ewm(span=14, adjust=False).mean()
    
    # Difference between the current closing price and the 14-day EMA
    df['diff_close_14day_ema'] = df['close'] - df['14day_ema_close']
    
    # Count the number of days in the past month where the closing price was above the opening price
    df['up_days'] = (df['close'] > df['open']).rolling(window=20).sum()
    
    # Ratio of up-days to total days
    df['up_day_ratio'] = df['up_days'] / 20
    
    # Highest high and lowest low over the last 20 days
    df['20day_highest_high'] = df['high'].rolling(window=20).max()
    df['20day_lowest_low'] = df['low'].rolling(window=20).min()
    
    # Range percentage
    df['range_percentage'] = (df['20day_highest_high'] - df['20day_lowest_low']) / df['20day_lowest_low']
    
    # Sum of volume on days when the closing price is higher than the opening price
    df['volume_up_days'] = df.apply(lambda row: row['volume'] if row['close'] > row['open'] else 0, axis=1)
    df['total_volume'] = df['volume'].rolling(window=20).sum()
    df['buying_pressure_ratio'] = df['volume_up_days'].rolling(window=20).sum() / df['total_volume']
    
    # Cumulative return over the past 30 trading days
