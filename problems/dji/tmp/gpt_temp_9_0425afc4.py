import pandas as pd

def heuristics_v2(df):
    # Calculate a simple moving average for the close prices over 10 and 30 days
    sma_10 = df['close'].rolling(window=10).mean()
    sma_30 = df['close'].rolling(window=30).mean()
    
    # Calculate a weighted moving average for the volumes over 5 days
    wma_5_volume = df['volume'].rolling(window=5).apply(lambda x: (x * (pd.Series(range(1,6)))) / 15, raw=True)
    
    # Custom momentum calculation over 10 days
    momentum_10 = df['close'] - df['close'].shift(10)
    
    # Combine factors into a single alpha factor using a weighted approach
    factor_value = (sma_10 + wma_5_volume + momentum_10) / 3
    
    # Apply a filter to remove noise; here, we simply cap and floor the factor
    heuristics_matrix = factor_value.clip(lower=-2, upper=2)
    
    return heuristics_matrix
