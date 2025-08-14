import pandas as pd

def heuristics_v2(df):
    def custom_heuristic(rolling_mean, rolling_std, weights):
        return (weights[0] * rolling_mean + weights[1] * rolling_std).mean(axis=1)
    
    rolling_window = 20
    weights = [0.5, 0.5]
    heuristics_matrix = df.rolling(window=rolling_window).agg(['mean', 'std']).apply(lambda x: custom_heuristic(x['mean'], x['std'], weights), axis=1)
    return heuristics_matrix
