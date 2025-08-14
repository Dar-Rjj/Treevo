def heuristics_v2(df):
    # Calculate the rate of change in close price over the last 5 days
    roc_5 = df['close'].pct_change(periods=5)
    # Calculate the rate of change in close price over the last 20 days
    roc_20 = df['close'].pct_change(periods=20)
    # Calculate the Average True Range (ATR) over the last 10 days
    tr = df[['high', 'low']].rolling(window=10).max(axis=1) - df[['high', 'low']].rolling(window=10).min(axis=1)
    atr_10 = tr.rolling(window=10).mean()
    # Combine factors into a single heuristics score
    heuristics_matrix = (roc_5 + roc_20 + atr_10) / 3
    return heuristics_matrix
