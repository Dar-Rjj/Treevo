import pandas as pd

    # Calculate the Average True Range (ATR)
    df['TR1'] = abs(df['high'] - df['low'])
    df['TR2'] = abs(df['high'] - df['close'].shift())
    df['TR3'] = abs(df['low'] - df['close'].shift())
    df['TrueRange'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()

    # Calculate the percentage change in volume over 20 days
    df['Volume_Change'] = (df['volume'] / df['volume'].shift(20) - 1) * 100

    # Combine ATR to close ratio and Volume Change
    df['Heuristic_Factor'] = (df['ATR'] / df['close']) + df['Volume_Change']

    # Smooth the heuristic factor using a Gaussian window
    heuristics_matrix = df['Heuristic_Factor'].rolling(window=18, win_type='gaussian').mean(std=3)

    return heuristics_matrix
