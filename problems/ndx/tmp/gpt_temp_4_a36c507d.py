def heuristics_v2(df):
    # Momentum-based Alpha Factors
    df['50_day_SMA'] = df['close'].rolling(window=50).mean()
    df['200_day_SMA'] = df['close'].rolling(window=200).mean()
    df['SMA_Cross_Over'] = df['50_day_SMA'] - df['200_day_SMA']
    
    df['ROC_14'] = df['close'].pct_change(periods=14)
    
    gain = df['close'].diff().apply(lambda x: x if x > 0 else 0)
    loss = df['close'].diff().apply(lambda x: -x if x < 0 else 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Volatility-based Alpha Factors
    df['daily_returns'] = df['close'].pct_change()
