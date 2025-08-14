import pandas as pd

def heuristics_v2(df):
    close_prices = df['close']
    
    wma_21_close = close_prices.rolling(window=21).apply(lambda x: (x * pd.Series(range(1, len(x) + 1))).sum() / pd.Series(range(1, len(x) + 1)).sum(), raw=False)
    wma_60_close = close_prices.rolling(window=60).apply(lambda x: (x * pd.Series(range(1, len(x) + 1))).sum() / pd.Series(range(1, len(x) + 1)).sum(), raw=False)
    std_5_close = close_prices.rolling(window=5).std()
    
    heuristics_matrix = (wma_21_close - wma_60_close) * std_5_close
    
    return heuristics_matrix
