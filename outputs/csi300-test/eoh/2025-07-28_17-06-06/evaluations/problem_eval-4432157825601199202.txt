importance to recent and notable price movements.}

```python
def heuristics_v2(df):
    df['10d_mean_close'] = df['close'].rolling(window=10).mean()
    df['abs_dist_to_10d_mean'] = abs(df['close'] - df['10d_mean_close'])
    total_weight = df['abs_dist_to_10d_mean'].rolling(window=20).sum()
    df['weighted_volume'] = df['volume'] * df['abs_dist_to_10d_mean']
    df['weighted_amount'] = df['amount'] * df['abs_dist_to_10d_mean']
    df['weighted_avg'] = (df['weighted_volume'].rolling(window=20).sum() + df['weighted_amount'].rolling(window=20).sum()) / total_weight
    heuristics_matrix = df['weighted_avg'].dropna()
    return heuristics_matrix
