def heuristics_v2(df):
    # Calculate moving averages of close prices
    df['5_day_ma_close'] = df['close'].rolling(window=5).mean()
    df['10_day_ma_close'] = df['close'].rolling(window=10).mean()
    df['20_day_ma_close'] = df['close'].rolling(window=20).mean()
    
    # Calculate momentum of close prices
    df['10_day_momentum'] = df['close'].pct_change(periods=10)
    df['20_day_momentum'] = df['close'].pct_change(periods=20)
    df['50_day_momentum'] = df['close'].pct_change(periods=50)
    
    # Calculate standard deviation of daily returns
