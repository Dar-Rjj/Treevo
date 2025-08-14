import pandas as pd
    
    def vwap(price, volume):
        return (price * volume).cumsum() / volume.cumsum()

    def rsi(series, periods=14):
        delta = series.diff(1)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.rolling(window=periods).mean()
        roll_down = down.abs().rolling(window=periods).mean()
        RS = roll_up / roll_down
        return 100.0 - (100.0 / (1.0 + RS))

    vwap_signal = vwap(df['close'], df['volume'])
    rsi_high = rsi(df['high'])
    combined_factor = (vwap_signal + rsi_high).rename('combined_factor')
    heuristics_matrix = combined_factor.ewm(span=30, adjust=False).mean().rename('heuristic_factor')

    return heuristics_matrix
