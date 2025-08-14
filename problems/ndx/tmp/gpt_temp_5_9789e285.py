import pandas as pd

def heuristics_v2(df):
    def calculate_momentum(series, window=15):
        return series.rolling(window=window).std() / series.rolling(window=window).mean()
    
    def calculate_log_return(series):
        return (series / series.shift(1)).apply(np.log).cumsum()

    momentum = calculate_momentum(df['close'])
    log_return = calculate_log_return(df['close'])

    heuristics_matrix = (momentum * 0.6 + log_return * 0.4).dropna()
    return heuristics_matrix
