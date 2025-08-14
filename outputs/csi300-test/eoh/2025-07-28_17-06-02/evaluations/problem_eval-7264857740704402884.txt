import pandas as pd

def heuristics_v2(df):
    def adaptive_lookback_volatility(row, short_window=10, long_window=30):
        short_vol = df['close'].rolling(window=short_window).std().iloc[-1]
        long_vol = df['close'].rolling(window=long_window).std().iloc[-1]
        return short_window if short_vol > long_vol else long_window
    
    df['amo'] = 0
    for i in range(len(df)):
        window = adaptive_lookback_volatility(df.iloc[:i+1])
        if i >= window:
            df.loc[df.index[i], 'amo'] = df.loc[df.index[i], 'close'] - df.loc[df.index[i-window], 'close']
    
    heuristics_matrix = df['amo'].copy()
    return heuristics_matrix
