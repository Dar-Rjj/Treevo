import pandas as pd
import numpy as np

def heuristics_v2(df):
    def compute_statistics(window):
        return pd.Series({
            'mean': window.mean(),
            'median': window.median(),
            'std': window.std()
        })

    roll_stats = df.rolling(window=20).agg(compute_statistics)
    weights = np.array([0.4, 0.3, 0.3])  # example weights for mean, median, std
    heuristics_matrix = (roll_stats * weights).sum(axis=1)
    heuristics_matrix = np.tanh(heuristics_matrix)  # non-linear transformation
    return heuristics_matrix
