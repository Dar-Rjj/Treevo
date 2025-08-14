import pandas as pd

def heuristics_v2(df):
    log_returns = df['close'].apply(lambda x: np.log(x)).diff().fillna(0)
    momentum = df['close'] - df['close'].shift(20)
    volatility = df['close'].rolling(window=20).std()
    heuristics_matrix = (log_returns * 0.4) + (momentum * 0.3) + (volatility * 0.3)
    return heuristics_matrix
