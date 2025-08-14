import pandas as pd
    import numpy as np
    
    # Calculate differences between high and low prices
    hl_diff = df['high'] - df['low']
    # Compute the moving average of the close price over a 5-day window
    close_ma_5 = df['close'].rolling(window=5).mean()
    # Calculate the ratio of today's volume to the 10-day exponential moving average of volume
    vol_ema_10 = df['volume'].ewm(span=10, adjust=False).mean()
    vol_ratio = df['volume'] / vol_ema_10
    # Generate the momentum factor by computing the return over the last 20 days
    momentum_20 = df['close'].pct_change(periods=20)
    
    # Construct the heuristics matrix from the calculated features
    heuristics_matrix = pd.DataFrame({
        'hl_diff': hl_diff,
        'close_ma_5': close_ma_5,
        'vol_ratio': vol_ratio,
        'momentum_20': momentum_20
    }).dropna().apply(lambda x: x * (x > 0), axis=0)  # Zero out negative values for interpretability

    return heuristics_matrix
