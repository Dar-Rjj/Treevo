import pandas as pd
    def RSI(series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def EMA(series, span=10):
        return series.ewm(span=span, adjust=False).mean()

    df['RSI'] = RSI(df['close'])
    df['Short_EMA'] = EMA(df['close'], 12)
    df['Long_EMA'] = EMA(df['close'], 26)
    df['Heuristic_Factor'] = ((df['Short_EMA'] - df['Long_EMA']) / df['close']) * df['RSI']
    
    heuristics_matrix = df['Heuristic_Factor']
    return heuristics_matrix
