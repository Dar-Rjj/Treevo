def heuristics_v2(df):
    # Intraday Momentum
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
