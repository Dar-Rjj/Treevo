def heuristics_v2(df):
    # Calculate the volume-adjusted moving average of open and close prices
    adjusted_ma = ((df['open'] + df['close']) / 2 * df['volume']).cumsum() / df['volume'].cumsum()
    # Calculate the exponentially weighted moving average of high and low prices
    ewma_price = (df['high'] + df['low']) / 2.ewm(span=10).mean()
    # Calculate the factor as the ratio of volume-adjusted moving average to EWMA
    factor_values = adjusted_ma / ewma_price
    # Apply a simple moving average for smoothing the factor values
    smoothed_factor = factor_values.rolling(window=10).mean()
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
