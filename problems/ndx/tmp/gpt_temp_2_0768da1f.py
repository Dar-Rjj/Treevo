def heuristics_v2(df):
    def calculate_daily_returns(series):
        return series.pct_change()
    
    daily_returns = df['close'].apply(calculate_daily_returns)
    short_term_mavg = daily_returns.rolling(window=5).mean()
    long_term_mavg = daily_returns.rolling(window=20).mean()
    heuristics_matrix = short_term_mavg - long_term_mavg
    return heuristics_matrix
