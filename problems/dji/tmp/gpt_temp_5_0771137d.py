import pandas as pd

def heuristics_v2(df):
    def rsi(series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def roc(series, period=12):
        return (series - series.shift(period)) / series.shift(period) * 100
    
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['RSI_14'] = rsi(df['close'])
    df['ROC_12'] = roc(df['close'])
    heuristics_matrix = df[['SMA_5', 'SMA_10', 'SMA_20', 'RSI_14', 'ROC_12']].copy()
    return heuristics_matrix
