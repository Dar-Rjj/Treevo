import pandas as pd
    import numpy as np
    from ta.momentum import RSIIndicator

    # Calculate short and long term weighted moving averages of the close price
    wma_short = df['close'].ewm(span=5, adjust=False).mean()
    wma_long = df['close'].ewm(span=20, adjust=False).mean()

    # Compute the RSI for the closing prices with a 14-day window
    rsi = RSIIndicator(close=df['close'], window=14).rsi()

    # Create a heuristic based on the difference between short and long WMA, adjusted by the RSI value
    heuristics_matrix = (wma_short - wma_long) * (70 - rsi)

    return heuristics_matrix
