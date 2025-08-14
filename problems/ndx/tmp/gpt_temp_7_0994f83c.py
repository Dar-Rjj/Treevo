def heuristics_v2(df):
    def calculate_roc(column, n):
        return (column - column.shift(n)) / column.shift(n)

    # Calculate ROC over a 50-day period
    roc = calculate_roc(df['close'], 50)
    
    # Calculate 20-day EMA of the closing price
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate 10-day and 50-day EMA
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_50 = df['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate RS (Relative Strength) as the ratio of 10-day EMA to 50-day EMA
    rs = ema_10 / ema_50
    
    # Calculate the ATR (Average True Range) over a 14-day period
    tr = pd.DataFrame({'H-L':df['high']-df['low'], 'H-Cp':abs(df['high']-df['close'].shift(1)), 
                       'L-Cp':abs(df['low']-df['close'].shift(1))})
    atr = tr.max(axis=1).rolling(window=14).mean()
    
    # Adjust the heuristics matrix with momentum, RS, and ATR
    heuristics_matrix = roc + (ema_20 - df['close']) + rs - atr
    
    return heuristics_matrix
