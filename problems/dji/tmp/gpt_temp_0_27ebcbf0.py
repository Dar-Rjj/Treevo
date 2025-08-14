def heuristics_v2(df):
    # Calculate ADL
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    adl = money_flow_volume.cumsum()

    # Calculate 30-day EMA of ADL
    adl_ema = adl.ewm(span=30, adjust=False).mean()

    # Calculate TR
    tr1 = abs(df['high'] - df['low'])
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate 21-day SD of TR
    sd_tr = true_range.rolling(window=21).std()

    # Create the heuristic matrix
    heuristics_matrix = (adl_ema / sd_tr).fillna(0)
    return heuristics_matrix
