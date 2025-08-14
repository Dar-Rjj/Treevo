import pandas as pd

def heuristics_v2(df):
    close_prices = df['close']
    volumes = df['volume']
    
    # Calculate the 20-day moving average of closing prices
    ma_20_close = close_prices.rolling(window=20).mean()
    
    # Calculate the 5-day rate of change for volume
    roc_5_volume = (volumes / volumes.shift(5)) - 1
    
    # Generate the heuristic value: (Close/Moving Average Close) - ROC Volume
    heuristics_values = (close_prices / ma_20_close) - roc_5_volume
    
    return heuristics_matrix
