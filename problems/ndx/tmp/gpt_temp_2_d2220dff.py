def heuristics_v2(df):
    open_weight = 0.5
    close_weight = 0.5
    rsi_period = 14
    ema_period = 9

    weighted_price = df['open'] * open_weight + df['close'] * close_weight
    delta = weighted_price.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    smoothed_rsi = rsi.ewm(span=ema_period, adjust=False).mean().dropna()
    
    return heuristics_matrix
