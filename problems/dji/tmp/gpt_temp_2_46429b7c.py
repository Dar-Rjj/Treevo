import pandas as pd
    heuristics_matrix = (df['volume'] / (df['high'] * df['low'])).ewm(span=5, adjust=False).mean()
    return heuristics_matrix
