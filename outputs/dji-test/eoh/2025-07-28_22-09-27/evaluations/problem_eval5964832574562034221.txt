def heuristics_v2(df):
    # Calculate the log difference of closing prices for trend strength
    log_returns = (df['close']).pct_change().apply(lambda x: np.log(1 + x)).fillna(0)
    
    # Calculate the average true range as a measure of volatility
    tr = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
    atr = tr.rolling(window=10).mean().fillna(0)
    
    # Calculate liquidity as the ratio of volume to the ATR
    liquidity = df['volume'] / atr
    
    # Combine factors into a heuristics matrix using a weighted sum
    heuristics_matrix = (log_returns * 0.6) + (liquidity * 0.4)
    
    return heuristics_matrix
