import pandas as pd

def heuristics_v2(df):
    # Calculate the 20-day average close
    avg_close_20 = df['close'].rolling(window=20).mean()
    # Calculate simple momentum
    simple_momentum = df['close'] - avg_close_20
    # Apply EMA for smoothing
    ema_smoothed = simple_momentum.ewm(span=10, adjust=False).mean()
    return heuristics_matrix
