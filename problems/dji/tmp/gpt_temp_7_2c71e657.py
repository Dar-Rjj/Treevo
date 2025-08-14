def heuristics_v2(df):
    # Calculate the difference between today's close price and yesterday's close price
    df['daily_return'] = df['close'].pct_change()
