importance to days with higher trading volume.}
```python
def heuristics_v2(df):
    df['5d_high'] = df['high'].rolling(window=5).max()
    df['5d_low'] = df['low'].rolling(window=5).min()
    df['rel_dist_to_5d_max'] = (df['5d_high'] - df['close']) / (df['5d_high'] - df['5d_low'])
    df['rel_dist_to_5d_min'] = (df['close'] - df['5d_low']) / (df['5d_high'] - df['5d_low'])
    df['volume_weighted_rel_dist'] = (df['rel_dist_to_5d_max'] + df['rel_dist_to_5d_min']) / 2 * df['volume']
    df['avg_vol_weighted_rel_dist_ema'] = df['volume_weighted_rel_dist'].ewm(span=5, adjust=False).mean() / df['volume'].ewm(span=5, adjust=False).mean()
    heuristics_matrix = df['avg_vol_weighted_rel_dist_ema'].dropna()
    return heuristics_matrix
