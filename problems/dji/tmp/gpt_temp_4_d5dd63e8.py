import pandas as pd

def heuristics_v2(df):
    heuristics_matrix = pd.Series(index=df.index)
    for i in range(10, len(df)):
        recent_return = (df['close'].iloc[i] - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
        medium_term_return = (df['close'].iloc[i] - df['close'].iloc[i-20]) / df['close'].iloc[i-20]
        long_term_return = (df['close'].iloc[i] - df['close'].iloc[i-60]) / df['close'].iloc[i-60]
        volume_change = (df['volume'].iloc[i] - df['volume'].iloc[i-10]).clip(lower=0)
        heuristics_matrix.iloc[i] = 0.5*recent_return + 0.3*medium_term_return + 0.2*long_term_return + 0.05 * volume_change
    return heuristics_matrix
