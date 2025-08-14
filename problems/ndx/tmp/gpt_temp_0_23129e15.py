import pandas as pd

def heuristics_v2(df):
    # Calculate the daily return
    df['Return'] = df['close'].pct_change()
    
    # Compute the exponential weighted moving average (EWMA) of returns
    ewma_return = df['Return'].ewm(span=20, adjust=False).mean()
    
    # Calculate the volume ratio
    avg_volume = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'] / avg_volume
    
    # Adjust EWMA return by volume ratio
    adjusted_ewma_return = ewma_return * volume_ratio
    
    # Shift the original return to align with the factors for prediction
    df['Future_Return'] = df['Return'].shift(-1)
    
    # Drop rows with NaN values resulting from the shift
    df = df.dropna()
    
    # Select the relevant columns and compute the heuristic factor
    heuristics_matrix = adjusted_ewma_return[df.index.isin(df.index)]
    
    return heuristics_matrix
