def heuristics_v2(df):
    # Calculate the ROC for a 10-day period
    roc = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100

    # Calculate the ADX for a 14-day period
    tr = pd.DataFrame()
    tr['h-l'] = df['high'] - df['low']
    tr['h-pc'] = abs(df['high'] - df['close'].shift(1))
    tr['l-pc'] = abs(df['low'] - df['close'].shift(1))
    tr['tr'] = tr.max(axis=1)
    atr = tr['tr'].rolling(window=14).mean()

    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff(-1)
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    smoothed_plus_dm = plus_dm.rolling(window=14).mean()
    smoothed_minus_dm = minus_dm.rolling(window=14).mean()

    plus_di = 100 * (smoothed_plus_dm / atr)
    minus_di = 100 * (smoothed_minus_dm / atr)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=14).mean()

    # Combine ROC and ADX into a single heuristics measure with weights
    heuristics_matrix = (0.5 * roc + 0.5 * adx)
    
    return heuristics_matrix
