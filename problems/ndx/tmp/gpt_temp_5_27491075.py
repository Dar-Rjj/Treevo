def heuristics_v2(df):
    df['Return'] = df['close'].pct_change()
    df['Future_Return'] = df['Return'].shift(-1)
    df = df.dropna()
    weights = pd.DataFrame(index=df.index, columns=df.columns[:-2], dtype='float64')
    for col in df.columns[:-2]:
        corr_with_return = df[col].rolling(window=20).corr(df['Future_Return'])
        mean = df[col].rolling(window=20).mean()
        weights[col] = (corr_with_return / mean)  # Adjust weights based on mean and correlation
    weights = weights.fillna(0)
    heuristics_matrix = (df[df.columns[:-2]] * weights).sum(axis=1)
    return heuristics_matrix
