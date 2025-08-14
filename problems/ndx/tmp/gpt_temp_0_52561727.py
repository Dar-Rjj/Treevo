import pandas as pd

def heuristics_v2(df):
    # Calculate the 10-day and 50-day simple moving averages of the close price
    sma_10 = df['close'].rolling(window=10).mean()
    sma_50 = df['close'].rolling(window=50).mean()
    
    # Calculate the ratio between the SMAs
    sma_ratio = sma_10 / sma_50
    
    # Calculate the True Range
    df['True Range'] = df[['high', 'low']].apply(lambda x: max(x['high'], x.shift(1)['close']) - min(x['low'], x.shift(1)['close']), axis=1)
    
    # Calculate the 30-day Average True Range (ATR)
    atr_30 = df['True Range'].rolling(window=30).mean()
    
    # Generate the heuristic matrix by multiplying the SMA ratio with the ATR
    heuristics_matrix = sma_ratio * atr_30
    
    return heuristics_matrix
