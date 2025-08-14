import pandas as pd

def heuristics_v2(df):
    # Calculate the simple moving average of the open price over a 10-day window
    sma_open = df['open'].rolling(window=10).mean()
    # Compute the momentum as the difference between the close price and the SMA of the open price
    momentum = df['close'] - sma_open
    # Adjust the momentum by the log of the volume to reflect trading intensity
    heuristics_matrix = momentum * (df['volume']).apply(lambda x: x if x == 0 else math.log(x + 1))
    return heuristics_matrix
