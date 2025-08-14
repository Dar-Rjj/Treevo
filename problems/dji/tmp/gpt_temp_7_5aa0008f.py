import pandas as pd
    # Calculate EMAs
    df['ema_short'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=30, adjust=False).mean()
    
    # EMA difference
    df['ema_diff'] = df['ema_short'] - df['ema_long']
    
    # Ratio of close to median of high and low
    df['price_ratio'] = df['close'] / ((df['high'] + df['low']) / 2)
    
    # Final heuristic matrix
    heuristics_matrix = df['ema_diff'] * df['price_ratio']
    
    return heuristics_matrix
