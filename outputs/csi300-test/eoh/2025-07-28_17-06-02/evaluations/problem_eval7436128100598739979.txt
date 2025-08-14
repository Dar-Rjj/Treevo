def heuristics_v2(df):
    # Kaufman Efficiency Ratio
    change = df['close'].diff().abs()
    volatility = (df['high'] - df['low']).abs()
    efficiency_ratio = change.rolling(window=10).sum() / volatility.rolling(window=10).sum()

    # On-Balance Volume (OBV)
    obv = (df['close'] > df['close'].shift(1)).astype(int) * df['volume'] - (df['close'] < df['close'].shift(1)).astype(int) * df['volume']
    obv = obv.cumsum()

    # Parabolic SAR
    sar = df['close'].copy()
    ep = df['close'].copy()
    acc_factor = 0.02
    max_acc_factor = 0.2
    trend = 1
    for i in range(1, len(df)):
        if trend == 1:
            sar.iloc[i] = sar.iloc[i-1] + acc_factor * (ep.iloc[i-1] - sar.iloc[i-1])
            if df['close'].iloc[i] < sar.iloc[i]:
                trend = -1
                sar.iloc[i] = ep.iloc[i-1]
                acc_factor = 0.02
        else:
            sar.iloc[i] = sar.iloc[i-1] + acc_factor * (ep.iloc[i-1] - sar.iloc[i-1])
            if df['close'].iloc[i] > sar.iloc[i]:
                trend = 1
                sar.iloc[i] = ep.iloc[i-1]
                acc_factor = 0.02
        ep.iloc[i] = df['close'].iloc[i] if trend == 1 and df['close'].iloc[i] > ep.iloc[i-1] else (df['close'].iloc[i] if trend == -1 and df['close'].iloc[i] < ep.iloc[i-1] else ep.iloc[i-1])
        acc_factor = min(acc_factor + 0.02, max_acc_factor)

    # Composite heuristic
    heuristics_matrix = (efficiency_ratio + obv + (df['close'] - sar)) / 3
    return heuristics_matrix
