import pandas as pd
    # Calculate the price change
    price_change = df['close'] - df['open']
    # Compute the ratio of price change to volume
    ratio = price_change / df['volume']
    # Calculate the EWMA of the ratio
    ewma_ratio = ratio.ewm(span=10, adjust=False).mean()
    heuristics_matrix = pd.Series(ewma_ratio, index=df.index)
    return heuristics_matrix
