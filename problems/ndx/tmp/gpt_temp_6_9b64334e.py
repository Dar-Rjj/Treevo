def heuristics_v2(df):
    # Calculate the 5-period Rate of Change (ROC) of the closing price
    roc_5 = df['close'].pct_change(periods=5)
    
    # Calculate the 14-period Average Directional Movement Index (ADX)
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = -df['low'].diff().clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(window=14).sum() / atr_14)
    minus_di = 100 * (minus_dm.rolling(window=14).sum() / atr_14)
    adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(window=14).mean()
    
    # Apply a custom heuristic to combine the ROC and ADX
    heuristics_matrix = (roc_5 + adx).rank(pct=True)
    
    return heuristics_matrix
