import pandas as pd

def heuristics_v2(df):
    def calculate_ema(df, column, span):
        return df[column].ewm(span=span, adjust=False).mean()

    def calculate_roc(df, column, window):
        return (df[column] - df[column].shift(window)) / df[column].shift(window)

    def dynamic_window(df, base_window, volatility_factor):
        std_dev = df['close'].rolling(window=base_window).std()
        return (base_window * std_dev / std_dev.rolling(window=base_window).mean()).fillna(base_window).astype(int)

    # Dynamic window based on recent volatility
    ema_window = dynamic_window(df, 10, 1)
    roc_window = dynamic_window(df, 5, 1)

    # Calculate factors
    ema_volume = calculate_ema(df, 'volume', ema_window)
    roc_close = calculate_roc(df, 'close', roc_window)

    # Combine factors
    heuristics_matrix = 0.6 * ema_volume + 0.4 * roc_close

    return heuristics_matrix
