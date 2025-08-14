import pandas as pd
    atr = (df['high'] - df['low']).rolling(window=14).mean()
    roc_amount = (df['amount'] / df['amount'].shift(1)) - 1
    heuristics_matrix = pd.Series(data=(atr.apply(lambda x: math.log(x + 1)) + roc_amount), index=df.index)
    return heuristics_matrix
