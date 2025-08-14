def heuristics_v2(df):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'] - df['close'].shift(1)
    
    # Compute 10-Day Exponential Moving Average of Daily Price Changes
    df['ema_10_daily_price_change'] = df['daily_price_change'].ewm(span=10, adjust=False).mean()
    
    # Calculate Intraday Return
    df['intraday_return'] = df['close'] - df['open']
