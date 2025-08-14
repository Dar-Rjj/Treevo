import pandas as pd
    
    # Calculate the 14-day Average True Range (ATR)
    df['H-L'] = df['high'] - df['low']
    df['H-Cp'] = abs(df['high'] - df['close'].shift(1))
    df['L-Cp'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # Calculate the 50-day Simple Moving Average (SMA) of the closing price
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Generate the heuristics factor
    df['Heuristic_Factor'] = (df['high'] - df['low']) / df['ATR_14'] + (df['close'] - df['SMA_50'])
    
    heuristics_matrix = df['Heuristic_Factor'].dropna()
    return heuristics_matrix
