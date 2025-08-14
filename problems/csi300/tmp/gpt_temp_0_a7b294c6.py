import pandas as pd
    
    # Calculate simple moving averages
    sma_5 = df['close'].rolling(window=5).mean()
    sma_20 = df['close'].rolling(window=20).mean()

    # Calculate momentum (price change over 10 days)
    momentum = df['close'] - df['close'].shift(10)

    # Custom heuristic: Blend moving average crossover and momentum
    heuristics_matrix = (sma_5 - sma_20) + momentum

    return heuristics_matrix
