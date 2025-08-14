importance to days with lower trading volumes and to cap the factor's value based on recent performance extremes.}

```python
def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Calculate the inverse of daily volumes for weighting
    volume_inverse = 1 / df['volume']
    # Calculate the weighted sum of daily returns
    weighted_return_sum = (daily_returns * volume_inverse).sum()
    # Calculate the maximum daily return over a 30-day period
    max_daily_return_30d = daily_returns.rolling(window=30).max().fillna(daily_returns.max())
    # Compute the heuristic factor
    heuristic_factor = weighted_return_sum / max_daily_return_30d if max_daily_return_30d > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series(heuristic_factor, index=df.index)
    return heuristics_matrix
