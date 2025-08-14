import pandas as pd

def heuristics_v2(df):
    # Calculate exponential moving averages
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_30 = df['close'].ewm(span=30, adjust=False).mean()
    ema_ratio = (ema_10 / ema_30) - 1
    
    # Momentum indicator
    momentum = df['close'] / df['close'].shift(20) - 1
    
    # Volatility measure using Average True Range (ATR)
    tr = df['high'] - df['low']
    tr['high_close'] = abs(df['high'] - df['close'].shift())
    tr['low_close'] = abs(df['low'] - df['close'].shift())
    tr['tr'] = tr[['high', 'high_close', 'low_close']].max(axis=1)
    atr = tr['tr'].rolling(window=14).mean()
    
    # Price range
    price_range = (df['high'] - df['low']) / df['close']
    
    # Volume weighted by close price
    volume_weighted = df['volume'] * df['close']
    
    # Modified Relative Strength Index (RSI) with smoothing
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Composite heuristic
    heuristics_matrix = (ema_ratio + momentum - atr + price_range + volume_weighted + rsi) / 6
    return heuristics_matrix
