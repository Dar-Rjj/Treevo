import numpy as np
def heuristics_v2(df):
    # Moving Averages
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    def hull_moving_average(data, window):
        half_window = int(window / 2)
        wma_half = data.rolling(window=half_window).mean()
        wma_full = data.rolling(window=window).mean()
        return (wma_half * 2 - wma_full).rolling(window=int(np.sqrt(window))).mean()
