import pandas as pd

def heuristics_v2(df):
    short_window = 10
    long_window = 50
    alpha = 0.3  # Smoothing factor for exponential weighting
    
    log_close = df['close'].apply(lambda x: np.log(x))
    short_ma = log_close.rolling(window=short_window).mean()
    long_ma = log_close.rolling(window=long_window).mean()
    ratio = short_ma / long_ma
    heuristics_matrix = ratio.ewm(alpha=alpha, adjust=False).mean().dropna()
    
    return heuristics_matrix
