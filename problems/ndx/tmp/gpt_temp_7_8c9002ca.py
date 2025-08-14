def heuristics_v2(df):
    # Calculate the 14-day Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate the 30-day average true range (ATR) as a custom volatility measure
    tr = pd.DataFrame({
        'h-l': df['high'] - df['low'],
        'h-pc': abs(df['high'] - df['close'].shift(1)),
        'l-pc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    atr_30 = tr.rolling(window=30).mean()
    
    # Adjust ATR by the trading range (high - low)
    range_adjusted_atr = atr_30 / (df['high'] - df['low'])
    
    # Combine RSI and range-adjusted ATR into a single factor
    combined_factor = rsi * range_adjusted_atr
    
    # Drop NaN values
    heuristics_matrix = combined_factor.dropna()
    
    return heuristics_matrix
