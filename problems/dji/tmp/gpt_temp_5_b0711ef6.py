import numpy as np
    
    # Calculate the log return
    df['log_return'] = np.log(df['close']) - np.log(df['open'])
    
    # 10-day moving average of the log return
    step1 = df['log_return'].rolling(window=10).mean()
    
    # 20-day exponentially weighted moving average of the volume
    step2 = df['volume'].ewm(span=20, adjust=False).mean()
    
    # Combine with weights
    heuristics_matrix = 0.6 * step1 + 0.4 * step2
    
    return heuristics_matrix
