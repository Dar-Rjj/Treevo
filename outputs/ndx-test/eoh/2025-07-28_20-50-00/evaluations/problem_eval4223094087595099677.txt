def heuristics_v2(df):
    # Calculate PPO
    ema_short = df['close'].ewm(span=12, adjust=False).mean()
    ema_long = df['close'].ewm(span=26, adjust=False).mean()
    ppo = ((ema_short - ema_long) / ema_long) * 100
    
    # Calculate ATR
    tr = pd.DataFrame({
        'h-l': df['high'] - df['low'],
        'h-pc': abs(df['high'] - df['close'].shift(1)),
        'l-pc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Composite heuristic
    heuristics_matrix = ppo + atr
    
    return heuristics_matrix
