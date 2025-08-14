def heuristics_v2(df):
    # Calculate Weighted Close-to-Open Return
    df['return_close_open'] = (df['close'] - df['open']) / df['open']
