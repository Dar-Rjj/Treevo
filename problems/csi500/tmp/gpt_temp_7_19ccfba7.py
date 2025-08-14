def heuristics_v2(df):
    # Calculate the difference between closing price and opening price
    df['daily_diff'] = df['close'] - df['open']
    
    # 5-day moving average of the daily price difference
    df['5d_ma_daily_diff'] = df['daily_diff'].rolling(window=5).mean()
    
    # 20-day standard deviation of the 5-day moving average of the daily price difference
    df['20d_std_5d_ma_daily_diff'] = df['5d_ma_daily_diff'].rolling(window=20).std()
    
    # Volatility-adjusted trend strength
    df['vol_adj_trend_strength'] = df['5d_ma_daily_diff'] / df['20d_std_5d_ma_daily_diff']
    
    # Count the number of days in the past month where the closing price was above the opening price
    df['up_days'] = (df['close'] > df['open']).rolling(window=20).sum()
    
    # Ratio of up-days to total days
    df['up_day_ratio'] = df['up_days'] / 20
    
    # High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']
    
    # Volume-Average Price
    df['volume_avg_price'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Volume-Averaged High-Low Spread Ratio
    df['vol_avg_high_low_spread_ratio'] = df['high_low_spread'] / df['volume_avg_price']
    
    # 14-day Exponential Moving Average (EMA) of the closing prices
    df['14d_ema_close'] = df['close'].ewm(span=14, adjust=False).mean()
    
    # Difference between the current closing price and the 14-day EMA
    df['close_14d_ema_diff'] = df['close'] - df['14d_ema_close']
    
    # Highest high and lowest low over the last 20 days
    df['highest_high_20d'] = df['high'].rolling(window=20).max()
    df['lowest_low_20d'] = df['low'].rolling(window=20).min()
    
    # Range percentage
    df['range_percentage'] = (df['highest_high_20d'] - df['lowest_low_20d']) / df['lowest_low_20d']
    
    # Sum of volume on days when the closing price is higher than the opening price, divided by the total volume
    df['buying_pressure'] = ((df['close'] > df['open']) * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Close-to-close price change
    df['daily_return'] = df['close'].pct_change()
