import pandas as pd

def heuristics_v2(df):
    window_size = 10
    df['sma_close'] = df['close'].rolling(window=window_size).mean()
    df['sma_high'] = df['high'].rolling(window=window_size).mean()
    df['roc_sma_close'] = df['sma_close'].pct_change()
    df['roc_sma_high'] = df['sma_high'].pct_change()
    df['sma_amount'] = df['amount'].rolling(window=window_size).mean()
    df['sma_volume'] = df['volume'].rolling(window=window_size).mean()
    df['rsi_close'] = df['close'].rolling(window=window_size).apply(lambda x: pd.Series(x).diff().to_numpy()[-window_size:].mean() / pd.Series(x).diff().abs().to_numpy()[-window_size:].mean(), raw=True)
    
    heuristics_matrix = (df['roc_sma_close'] + df['roc_sma_high']) * (df['sma_amount'] / df['sma_volume']) * df['rsi_close']
    return heuristics_matrix
