def heuristics_v2(df):
    # Calculate the True Range (TR)
    df['tr'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    
    # Calculate the Exponential Moving Average (EMA) of the True Range (TR) over 14 periods
    atr_ema_14 = df['tr'].ewm(span=14, adjust=False).mean()
    
    # Calculate the weighted sum of volumes, weights decrease linearly as the date moves away from the current one
    weights = pd.Series(range(1, 15), index=df.index[-14:])
    vol_weighted_sum = df['volume'].rolling(window=14).apply(lambda x: (x * weights).sum(), raw=False)
    
    # Factor calculation: EMA of ATR divided by the weighted sum of volumes
    heuristics_matrix = atr_ema_14 / vol_weighted_sum
    
    return heuristics_matrix
