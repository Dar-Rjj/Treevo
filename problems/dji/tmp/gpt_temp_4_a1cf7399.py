def heuristics_v2(df):
    def vwap(df):
        return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    def log_return(series, periods=1):
        return (series / series.shift(periods)).apply(np.log)

    vw = vwap(df)
    log_ret_high = log_return(df['high'])
    log_ret_low = log_return(df['low'])
    price_to_vwap_ratio = df['close'] / vw
    combined_factor = (price_to_vwap_ratio + log_ret_high - log_ret_low).rename('combined_factor')
    heuristics_matrix = combined_factor.ewm(span=20, adjust=False).mean().rename('heuristic_factor')

    return heuristics_matrix
