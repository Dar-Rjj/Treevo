import pandas as pd

    def compute_ema(window, df):
        return df['close'].ewm(span=window, adjust=False).mean()

    def compute_rsi(window, df):
        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['ema_10'] = compute_ema(10, df)
    df['rsi_5'] = compute_rsi(5, df)
    heuristics_matrix = (df['ema_10'] - df['close']) / df['close'] + df['rsi_5']
    return heuristics_matrix
