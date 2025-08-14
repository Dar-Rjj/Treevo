import pandas as tf

    def calculate_rsi(column, n=14):
        delta = column.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.ewm(com=n-1, adjust=False).mean()
        roll_down = down.ewm(com=n-1, adjust=False).mean().abs()
        rsi = 100.0 - (100.0 / (1.0 + roll_up / roll_down))
        return rsi
    
    # Calculate RSI over a 14-day period
    rsi = calculate_rsi(df['close'])
    
    # Calculate 20-day exponential moving average of the closing price
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate Average True Range over 14 days
    tr = tf.DataFrame({'high': df['high'], 'low': df['low'], 'prev_close': df['close'].shift(1)})
    tr['tr'] = tf.max([(tr['high'] - tr['low']), abs(tr['high'] - tr['prev_close']), abs(tr['low'] - tr['prev_close'])], axis=0)
    atr_14 = tr['tr'].rolling(window=14).mean()
    
    # Adjust the heuristics matrix with RSI, EMA, and ATR
    heuristics_matrix = rsi + (df['close'] - ema_20) - atr_14
    
    return heuristics_matrix
