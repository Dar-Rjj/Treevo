import pandas as pd

def heuristics_v2(df):
    def calculate_true_range(row):
        return max(row['high'] - row['low'], abs(row['high'] - row['close'].shift(1)), abs(row['low'] - row['close'].shift(1)))
    
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['true_range'] = df.apply(calculate_true_range, axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()
    df['rsi'] = calculate_rsi(df['close'])
    heuristics_matrix = df['rsi'] * df['atr']
    return heuristics_matrix
