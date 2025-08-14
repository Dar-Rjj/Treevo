def heuristics_v2(df):
    def compute_signal(row):
        price_change = row['close'] - row['open']
        volume_impact = (row['volume'] - df.shift(1).loc[row.name, 'volume']) / df.shift(1).loc[row.name, 'volume']
        return price_change * volume_impact if volume_impact != 0 else 0

    signals = df.apply(compute_signal, axis=1)
    heuristics_matrix = pd.Series(signals, index=df.index, name='signal')
    return heuristics_matrix
