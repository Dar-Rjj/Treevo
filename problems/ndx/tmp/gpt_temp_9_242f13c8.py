import pandas as pd

def heuristics_v2(df):
    df['Return'] = df['close'].pct_change()
    df['Future_Return'] = df['Return'].shift(-1)
    df = df.dropna()
    weights = pd.DataFrame(index=df.index, columns=df.columns[:-2], dtype='float64')
    for col in df.columns[:-2]:
        corr_with_return = df[col].rolling(window=20).corr(df['Future_Return'])
        mad = df[col].rolling(window=20).apply(lambda x: x.mad(), raw=True)
        skew = df[col].rolling(window=20).skew()
        weights[col] = (corr_with_return / mad) * skew
    weights = weights.fillna(0)
    heuristics_matrix = (df[df.columns[:-2]] * weights).sum(axis=1)
    return heuristics_matrix
