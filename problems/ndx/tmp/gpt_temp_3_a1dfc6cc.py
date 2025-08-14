import pandas as pd
    
    # Calculate the daily return
    daily_return = df['close'].pct_change()
    
    # Calculate the Average True Range (ATR) over a 14-day period
    df['tr0'] = abs(df["high"] - df["low"])
    df['tr1'] = abs(df["high"] - df["close"].shift())
    df['tr2'] = abs(df["low"] - df["close"].shift())
    tr = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Calculate the Median Absolute Deviation (MAD) of daily returns
    mad = daily_return.rolling(window=20).apply(lambda x: (x - x.median()).abs().median(), raw=False)
    
    # Construct the factor
    heuristics_matrix = atr / mad

    return heuristics_matrix
