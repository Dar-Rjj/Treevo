def heuristics_v2(df):
    def roc(series, n=14):
        return (series / series.shift(n) - 1) * 100
    
    def modified_adl(df):
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        money_flow_volume = money_flow_multiplier * df['volume'].apply(lambda x: max(0.0001, x)).apply(np.log)
        return money_flow_volume.cumsum()
    
    close_roc = roc(df['close'])
    mod_adl_line = modified_adl(df)
    heuristics_matrix = (close_roc + mod_adl_line) / 2
    return heuristics_matrix
