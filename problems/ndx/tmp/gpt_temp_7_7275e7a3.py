def heuristics_v2(df):
    # Calculate short and long term exponentially weighted moving averages for the closing price
    short_term_ewma = df['close'].ewm(span=21, adjust=False).mean()
    long_term_ewma = df['close'].ewm(span=63, adjust=False).mean()

    # Calculate the log returns
    log_returns = np.log(df['close'] / df['close'].shift(1))

    # Calculate the volume-weighted average of high, low, and close prices
    avg_price = (df['high'] + df['low'] + df['close']) * df['volume']
    avg_price = avg_price / df['volume']

    # Calculate the modified volatility using the standard deviation of log returns
    vol = log_returns.rolling(window=28).std() * np.sqrt(252)

    # Combine the EWMA difference and the modified volatility
    ewma_diff = short_term_ewma - long_term_ewma
    heuristics_matrix = (ewma_diff + vol) / 2

    return heuristics_matrix
