import pandas as pd
    import talib

    # Calculate ADX, +DI, -DI
    adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Calculate RSI
    rsi = talib.RSI(df['close'], timeperiod=14)
    
    # Ratio of +DI to -DI
    di_ratio = plus_di / minus_di
    
    # Alpha factor as a combination of ADX, DI ratio, and RSI
    heuristics_matrix = (adx * di_ratio) * rsi
    
    return heuristics_matrix
