import pandas as pd

def heuristics_v2(df):
    def compute_rsi(series, window=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    ema_low = df['low'].ewm(span=20, adjust=False).mean()
    rsi_close = compute_rsi(df['close'])
    log_return = df['close'].apply(lambda x: np.log(x)) - df['close'].shift(1).apply(lambda x: np.log(x))
    hl_ratio = df['high'] / ema_low
    heuristics_matrix = rsi_close + log_return + hl_ratio
    return heuristics_matrix
