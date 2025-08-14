def heuristics_v2(df):
    # Calculate the 12-day and 26-day exponential moving averages for the closing price
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()

    # Calculate the Percentage Price Oscillator (PPO)
    ppo = ((ema_12 - ema_26) / ema_26) * 100

    # Calculate the Average True Range (ATR) over a 14-day period
    tr = pd.DataFrame(index=df.index)
    tr['h-l'] = df['high'] - df['low']
    tr['h-pc'] = abs(df['high'] - df['close'].shift())
    tr['l-pc'] = abs(df['low'] - df['close'].shift())
    tr['tr'] = tr.max(axis=1)
    atr = tr['tr'].rolling(window=14).mean()

    # Combine PPO and ATR into a single heuristics measure
    heuristics_matrix = (ppo + atr) / 2

    return heuristics_matrix
