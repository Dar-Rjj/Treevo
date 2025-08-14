def heuristics_v2(df):
    window = 20
    df['Positive_Returns'] = (df['close'].pct_change() > 0).rolling(window=window).sum()
    df['Negative_Returns'] = (df['close'].pct_change() < 0).rolling(window=window).sum()
    df['Volume_Log'] = np.log(df['volume'].rolling(window=window).mean())
    df['Alpha_Factor'] = (df['Positive_Returns'] / df['Negative_Returns']) * df['Volume_Log']
    heuristics_matrix = df['Alpha_Factor'].dropna()
    return heuristics_matrix
