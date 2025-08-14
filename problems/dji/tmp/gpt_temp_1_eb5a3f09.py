def heuristics_v2(df):
    def calc_factor(row):
        price_diff = (row['close'] - row['open']) * 0.5 + (row['high'] - row['low']) * 0.5
        momentum = (row['close'] - df['close'].shift(1).iloc[-1]) if pd.notna(df['close'].shift(1).iloc[-1]) else 0
        volume_adj = row['volume'] / df['volume'].mean()
        return (price_diff + momentum) * volume_adj

    heuristics_matrix = df.apply(calc_factor, axis=1)
    return heuristics_matrix
