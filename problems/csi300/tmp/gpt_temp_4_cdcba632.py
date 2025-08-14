import pandas as pd

def heuristics_v2(df):
    # Calculate weighted moving averages
    wma_7 = df['close'].rolling(window=7).apply(lambda x: (x * pd.Series(range(1, 8), index=x.index)).sum() / 28, raw=False)
    wma_14 = df['close'].rolling(window=14).apply(lambda x: (x * pd.Series(range(1, 15), index=x.index)).sum() / 105, raw=False)
    wma_ratio = (wma_7 / wma_14) - 1
    
    # Modified Relative Strength Index (RSI) with a 21-day window
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=21).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=21).mean()
    rs = gain / loss
    rsi_mod = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_value = (df['close'] - sma_20) / (2 * std_20)
    
    # Composite heuristic
    heuristics_matrix = (wma_ratio + rsi_mod + bb_value) / 3
    return heuristics_matrix
