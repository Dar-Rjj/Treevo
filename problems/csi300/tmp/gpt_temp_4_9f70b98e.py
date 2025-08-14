importance.}

```python
def heuristics_v2(df):
    close_roc = df['close'].pct_change(20)
    cum_return = (df['close'] / df['close'].shift(10)) - 1
    vol_ratio = df['volume'] / df['volume'].rolling(window=30).mean()
    
    # Relative importance weights
    weight_close_roc = 0.4
    weight_cum_return = 0.3
    weight_vol_ratio = 0.3
    
    heuristics_matrix = (weight_close_roc * close_roc + 
                         weight_cum_return * cum_return + 
                         weight_vol_ratio * vol_ratio).dropna()
    return heuristics_matrix
