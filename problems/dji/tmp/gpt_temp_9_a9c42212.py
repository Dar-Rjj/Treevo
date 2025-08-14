importance ranking technique.}

```python
def heuristics_v2(df):
    import pandas as pd
    import numpy as np
    
    # Define a few potential alpha factors
    alpha1 = (df['close'] - df['open']) / df['open']
    alpha2 = df['volume'].rolling(window=5).mean() - df['volume']
    alpha3 = (df['high'] - df['low']) / df['close']
    alpha4 = df['close'].pct_change()
    alpha5 = (df['close'] - df['open']) * df['volume']
    
    # Combine the alphas into a DataFrame
    alphas_df = pd.DataFrame({
        'alpha1': alpha1,
        'alpha2': alpha2,
        'alpha3': alpha3,
        'alpha4': alpha4,
        'alpha5': alpha5
    })
    
    # Use a simple scoring mechanism to select the best alphas (as an example, using correlation here)
    scores = alphas_df.corrwith(df['close'].pct_change().shift(-1)).abs()
    selected_alphas = scores[scores > 0.05].index  # Select only those with a correlation above a threshold
    
    # Construct the final heuristics matrix with selected alphas
    heuristics_matrix = alphas_df[selected_alphas]
    
    return heuristics_matrix
```
This function does not normalize any data as per the instructions and return heuristics_matrix
