import pandas as pd

    # Calculate Average True Range (ATR) over a 14-day period
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift())
    low_close_prev = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()

    # Calculate the percentage change in close prices
    close_pct_change = df['close'].pct_change()

    # Calculate 50-day Exponential Moving Average (EMA) of volume
    volume_ema = df['volume'].ewm(span=50, adjust=False).mean()
    
    # Compile heuristics into a matrix
    heuristics_matrix = atr + close_pct_change + volume_ema
    
    return heuristics_matrix
