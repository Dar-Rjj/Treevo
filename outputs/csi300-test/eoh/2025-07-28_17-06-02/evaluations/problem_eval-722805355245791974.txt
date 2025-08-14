import pandas as pd

def heuristics_v2(df):
    # Calculate exponential moving averages
    ema_7 = df['close'].ewm(span=7, adjust=False).mean()
    ema_14 = df['close'].ewm(span=14, adjust=False).mean()
    ema_ratio = (ema_7 / ema_14) - 1
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Average True Range (ATR)
    tr = df[['high'-'low', 'high'-'close'.shift(1), 'low'-'close'.shift(1)]].max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Volume and Close Price Momentum
    volume_momentum = df['volume'].pct_change(7)
    close_momentum = df['close'].pct_change(7)
    
    # Composite heuristic
    heuristics_matrix = (ema_ratio + rsi - atr + volume_momentum + close_momentum) / 5
    return heuristics_matrix
