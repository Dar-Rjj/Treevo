import pandas as pd

def heuristics_v2(df):
    # Calculate exponential moving averages
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_ratio = (ema_5 / ema_10) - 1
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Average True Range (ATR)
    tr = df[['high' - 'low', (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()]].max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Adjusted volume weighted by close price
    volume_weighted_adj = (df['volume'] * df['close']) / df['close'].rolling(window=5).mean()
    
    # Composite heuristic
    heuristics_matrix = (ema_ratio + rsi - atr + volume_weighted_adj) / 4
    return heuristics_matrix
