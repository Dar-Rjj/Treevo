import pandas as pd
    
    # Example heuristic: difference between high and low prices
    heuristic_1 = df['high'] - df['low']
    
    # Example heuristic: change in closing price
    heuristic_2 = df['close'].diff()
    
    # Example heuristic: volume relative to its 5-day moving average
    heuristic_3 = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Combine heuristics into a single DataFrame
    heuristics_matrix = pd.DataFrame({
        'heuristic_1': heuristic_1,
        'heuristic_2': heuristic_2,
        'heuristic_3': heuristic_3
    })
    
    # Sum the heuristics to create a final factor value
    heuristics_series = heuristics_matrix.sum(axis=1)
    
    return heuristics_matrix
