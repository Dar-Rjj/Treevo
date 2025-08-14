def heuristics_v2(df):
    # Calculate the modified Volume-Weighted Average Price (VWAP)
    vwap = (df['volume'] * (df['high'] + df['low']) / 2).cumsum() / df['volume'].cumsum()
    # Calculate the factor as the ratio of VWAP to close price
    factor_values = vwap / df['close']
    # Apply an exponential moving average for smoothing the factor values
    smoothed_factor = factor_values.ewm(span=10, adjust=False).mean()
    return heuristics_matrix
