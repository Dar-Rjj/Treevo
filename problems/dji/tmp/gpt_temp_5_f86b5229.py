import pandas as pd

def heuristics_v2(df):
    # Calculate OBV
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Calculate 30-day EMA of OBV
    obv_ema = obv.ewm(span=30, adjust=False).mean()

    # Calculate 21-day Standard Deviation of OBV
    std_obv = obv.rolling(window=21).std()

    # Create the heuristic matrix
    heuristics_matrix = (obv_ema / std_obv).fillna(0)
    return heuristics_matrix
