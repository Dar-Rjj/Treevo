import pandas as pd

def heuristics_v2(df):
    # Calculate the 30-day and 90-day weighted moving averages of the close price
    wma_30 = df['close'].rolling(window=30).apply(lambda x: (x * pd.Series(range(1, len(x)+1))).sum() / pd.Series(range(1, len(x)+1)).sum(), raw=False)
    wma_90 = df['close'].rolling(window=90).apply(lambda x: (x * pd.Series(range(1, len(x)+1))).sum() / pd.Series(range(1, len(x)+1)).sum(), raw=False)
    
    # Calculate the ratio between the WMAs
    wma_ratio = wma_30 / wma_90
    
    # Calculate the 40-day exponential moving average of the daily trading volume
    ema_vol_40 = df['volume'].ewm(span=40, adjust=False).mean()
    
    # Generate the heuristic matrix by multiplying the WMA ratio with the EMA of volume
    heuristics_matrix = wma_ratio * ema_vol_40
    
    return heuristics_matrix
