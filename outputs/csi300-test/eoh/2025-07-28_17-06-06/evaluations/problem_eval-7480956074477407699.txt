import pandas as pd
    df['return'] = df['close'].pct_change()
    df['high_low_diff'] = (df['high'] - df['low']) / df['close']
    df['momentum'] = df['return'].rolling(window=5).mean()
    heuristics_matrix = df['volume'] * (df['momentum'] + df['high_low_diff'])
    return heuristics_matrix
