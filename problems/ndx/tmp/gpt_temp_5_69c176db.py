importance to predict future returns.}

```python
import pandas as pd

def heuristics_v2(df):
    # Calculate the daily return
    df['Return'] = df['close'].pct_change()
    
    # Shift the return to align with the factors for prediction
    df['Future_Return'] = df['Return'].shift(-1)
    
    # Drop rows with NaN values resulting from the shift
    df = df.dropna()
    
    # Calculate the change in close price
    df['Close_Change'] = df['close'].diff()
    
    # Calculate the volume-weighted close change
    df['Volume_Weighted_Change'] = df['Close_Change'] * df['volume']
    
    # Compute the rolling sum of the volume-weighted change over a 20-day window
    df['Rolling_Volume_Sum'] = df['Volume_Weighted_Change'].rolling(window=20).sum()
    
    # Calculate the ratio of the current volume to the 20-day average volume
    df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Combine the factors
    heuristics_matrix = (0.5 * df['Rolling_Volume_Sum'] + 0.5 * df['Volume_Ratio'])
    
    return heuristics_matrix
