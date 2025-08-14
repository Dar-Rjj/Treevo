def heuristics_v2(df):
    window_size = 20
    adx_window = 14
    df['roc_close'] = df['close'].pct_change(window_size)
    high_minus_low = df['high'] - df['low']
    high_minus_prev_close = abs(df['high'] - df['close'].shift(1))
    low_minus_prev_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_minus_low, high_minus_prev_close, low_minus_prev_close], axis=1).max(axis=1)
    direction_move_pos = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
    direction_move_neg = -direction_move_pos
    direction_move_pos = direction_move_pos.ewm(alpha=1/adx_window, adjust=False).mean()
    direction_move_neg = direction_move_neg.ewm(alpha=1/adx_window, adjust=False).mean()
    adx = 100 * (abs(direction_move_pos - direction_move_neg) / (direction_move_pos + direction_move_neg)).ewm(alpha=1/adx_window, adjust=False).mean()
    heuristics_matrix = df['roc_close'] * adx
    return heuristics_matrix
