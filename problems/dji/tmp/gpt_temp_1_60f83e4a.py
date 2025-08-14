def heuristics_v2(df):
    # Calculate Base Intraday Return
    df['base_intraday_return'] = (df['close'] - df['open']) / df['open']
