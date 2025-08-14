def heuristics_v2(df):
    def macd_line(price, fast=12, slow=26):
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        return (ema_fast - ema_slow) / ema_slow  # Normalized MACD

    def log_return(series, periods=1):
        return np.log(series / series.shift(periods))

    macd_signal_normalized = macd_line(df['close'])
    log_ret = log_return(df['close'])
    combined_factor = (macd_signal_normalized + log_ret).rename('combined_factor')
    heuristics_matrix = combined_factor.ewm(span=50, adjust=False).mean().rename('heuristic_factor')

    return heuristics_matrix
