defined by the absolute difference between the daily closing price and its 10-day exponential moving average, to highlight significant price movements.}

```python
def heuristics_v2(df):
    df['10d_ema_close'] = df['close'].ewm(span=10, adjust=False).mean()
    df['abs_diff_to_10d_ema'] = (df['close'] - df['10d_ema_close']).abs()
    total_weight = df['abs_diff_to_10d_ema'].rolling(window=20).sum()
    df['weighted_volume'] = df['volume'] * df['abs_diff_to_10d_ema']
    df['weighted_amount'] = df['amount'] * df['abs_diff_to_10d_ema']
    df['weighted_sum'] = (df['weighted_volume'].rolling(window=20).sum() + df['weighted_amount'].rolling(window=20).sum())
    heuristics_matrix = df['weighted_sum'].dropna()
    return heuristics_matrix
