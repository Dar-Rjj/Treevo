import pandas as pd

def heuristics_v2(df):
    # Calculate the weighted moving average of the close price using sqrt(volume) as weights
    wma_close = (df['close'] * df['volume'].apply(np.sqrt)).rolling(window=20).sum() / df['volume'].apply(np.sqrt).rolling(window=20).sum()
    # Calculate the exponential moving average of the median between high and low prices
    ema_median_high_low = ((df['high'] + df['low']) / 2).ewm(span=20, adjust=False).mean()
    # Generate the heuristic factor by subtracting the EMA of median high/low from WMA of close
    heuristics_matrix = wma_close - ema_median_high_low
    return heuristics_matrix
