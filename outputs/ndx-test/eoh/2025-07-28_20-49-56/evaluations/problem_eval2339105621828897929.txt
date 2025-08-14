importance to price momentum and money flow.}

```python
def heuristics_v2(df):
    # Calculate the 10-day Rate of Change (ROC) for the closing price
    roc = df['close'].pct_change(periods=10)
    
    # Calculate the Money Flow Multiplier
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    
    # Calculate the Money Flow Volume
    mf_volume = mfm * df['volume']
    
    # Calculate the Accumulation/Distribution Line
    adl = mf_volume.cumsum()
    
    # Combine ROC and ADL into a single heuristics measure
    heuristics_matrix = (roc + adl) / 2
    
    return heuristics_matrix
