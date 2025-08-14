import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate percentage change for each column (feature)
    pct_change = df.pct_change()
    
    # Rolling window for pattern identification
    window_size = 30
    rolling_corr = {}
    
    # Target future return calculation
    future_return_days = 5
    df['future_return'] = df['close'].pct_change(future_return_days).shift(-future_return_days)
    
    # Drop NaN values created by future_return shift
    pct_change = pct_change.dropna()
    df = df.dropna()
    
    for col in pct_change.columns:
        # Calculate rolling correlation with future return
        rolling_corr[col] = pct_change[col].rolling(window=window_size).corr(df['future_return'])
        
    # Convert to DataFrame for easier manipulation
    corr_df = pd.DataFrame(rolling_corr)
    
    # Rank the correlations and pick the top n
    ranked_corr = corr_df.rank(axis=1, ascending=False)
    top_n = 3
    selected_features = ranked_corr.apply(lambda x: x <= top_n, axis=1)
    
    # Calculate the final heuristics matrix using selected features
    heuristics_matrix = (selected_features * pct_change).sum(axis=1)
    
    return heuristics_matrix
```

This function `heuristics_v2` takes a DataFrame of market data, including open, high, low, close, and volume, and returns a Series representing the calculated heuristic factors aiming at predicting future stock return heuristics_matrix
