import pandas as pd
    # Calculate the 14-day exponentially weighted moving average of the closing price
    ewma = df['close'].ewm(span=14, adjust=False).mean()
    
    # Compute the ratio of the current close price to the EWMA
    ewma_ratio = df['close'] / ewma
    
    # Calculate Aroon Oscillator components
    period = 14
    high_roll = df['high'].rolling(window=period)
    low_roll = df['low'].rolling(window=period)
    aroon_up = ((period - high_roll.apply(lambda x: list(x).index(max(x)))) / period) * 100
    aroon_down = ((period - low_roll.apply(lambda x: list(x).index(min(x)))) / period) * 100
    aroon_oscillator = aroon_up - aroon_down

    # Combine the EWMA ratio and the Aroon Oscillator to form the heuristic matrix
    heuristics_matrix = ewma_ratio * aroon_oscillator.fillna(0)
    
    return heuristics_matrix
