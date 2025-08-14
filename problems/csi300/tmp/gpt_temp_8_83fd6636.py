def heuristics_v2(df):
    # 3-Day Percentage Change in Closing Price
    pct_change = df['close'].pct_change(periods=3)
    
    # True Range (TR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Exponential Moving Average of True Range (EMA-TR) over 10 days
    ema_tr = true_range.ewm(span=10, adjust=False).mean()
    
    # 10-Day Simple Moving Average of Volume
    sma_volume = df['volume'].rolling(window=10).mean()
    
    # Volume Momentum: Ratio of today's volume to the 10-day SMA of volume
    volume_momentum = df['volume'] / sma_volume
    
    # Heuristics matrix combining 3-day percentage change, 10-day EMA-TR, and volume momentum
    heuristics_matrix = (pct_change + ema_tr) * volume_momentum
    return heuristics_matrix
