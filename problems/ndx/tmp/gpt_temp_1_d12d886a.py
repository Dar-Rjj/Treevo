importance to days with lower volume.}

```python
import pandas as pd

def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Calculate the inverse of daily volumes for weighting
    inv_volume = 1 / df['volume']
    # Adjust infinities due to division by zero in volumes
    inv_volume = inv_volume.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Calculate the weighted average of daily returns using the inverse of volumes as weights
    weighted_avg_daily_returns = (daily_returns * inv_volume).sum() / inv_volume.sum()
    # Calculate the standard deviation of the daily volumes
    volume_std = df['volume'].std()
    # Compute the heuristic factor
    heuristic_factor = weighted_avg_daily_returns / volume_std if volume_std > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
