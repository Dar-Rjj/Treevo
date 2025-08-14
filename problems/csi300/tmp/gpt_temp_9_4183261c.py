import pandas as pd

    # Calculate Exponential Moving Average for the close price
    ema_window = 12
    df['Close_EMA'] = df['close'].ewm(span=ema_window, adjust=False).mean()

    # Compute Relative Strength Index (RSI) based on the amount
    def compute_rsi(series, window=14):
        delta = series.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.rolling(window).mean()
        roll_down = down.abs().rolling(window).mean()
        RS = roll_up / roll_down
        return 100.0 - (100.0 / (1.0 + RS))
    
    df['Amount_RSI'] = compute_rsi(df['amount'])

    # Combine factors into a single heuristic
    heuristics_matrix = (df['Close_EMA'] - df['close']) * df['Amount_RSI']
    
    return heuristics_matrix
