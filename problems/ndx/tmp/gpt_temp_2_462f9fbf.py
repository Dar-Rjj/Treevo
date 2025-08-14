def heuristics_v2(df):
    # Momentum Indicators
    df['roc_close'] = df['close'].pct_change(10)  # Rate of change in closing price over 10 days
    df['pct_change_volume'] = df['volume'].pct_change(10)  # Percentage change in volume over 10 days
    
    # Volatility Indicators
    df['intraday_range'] = df['high'] - df['low']  # Intraday range
    df['std_daily_returns'] = df['close'].pct_change().rolling(window=10).std()  # Standard deviation of daily returns
