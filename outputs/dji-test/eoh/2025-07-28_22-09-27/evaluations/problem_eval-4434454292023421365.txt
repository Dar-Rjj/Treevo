import pandas as pd
    def compute_rsi(series, window=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        RS = gain / loss
        return 100 - (100 / (1 + RS))
    
    def compute_macd(series, n_fast=12, n_slow=26, n_signal=9):
        ema_fast = series.ewm(span=n_fast, min_periods=n_fast-1).mean()
        ema_slow = series.ewm(span=n_slow, min_periods=n_slow-1).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=n_signal, min_periods=n_signal-1).mean()
        return macd - signal
    
    close_prices = df['close']
    rsi_series = compute_rsi(close_prices)
    macd_series = compute_macd(close_prices)
    heuristics_matrix = (rsi_series + macd_series)/2
    return heuristics_matrix
