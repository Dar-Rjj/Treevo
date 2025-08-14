import numpy as np
    heuristics_matrix = (df['close'].pct_change() * np.log(df['volume'])).shift(-1).dropna()
    return heuristics_matrix
