def heuristics_v2(df):
    # Calculate the 5-day Rate of Change (ROC) of the closing price
    roc_close = df['close'].pct_change(periods=5)
    
    # Calculate the 14-day Average Directional Index (ADX)
    high_diff = df['high'] - df['high'].shift(1)
    low_diff = df['low'].shift(1) - df['low']
    plus_dm = high_diff.where(high_diff > 0, 0)
    minus_dm = low_diff.where(low_diff > 0, 0)
    tr = df['high'].combine(df['low'], max) - df['low'].combine(df['high'], min)
    tr = tr.rolling(window=14).sum()
    plus_di = 100 * (plus_dm.rolling(window=14).sum() / tr)
    minus_di = 100 * (minus_dm.rolling(window=14).sum() / tr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=14).mean()

    # Create a heuristic score by combining ROC and ADX
    heuristics_matrix = (roc_close * adx).dropna()
    
    return heuristics_matrix
