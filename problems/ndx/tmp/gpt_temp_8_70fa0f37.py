import numpy as np
def heuristics_v2(df):
    # Momentum Indicators
    n = 10
    df['Momentum_10'] = (df['close'] - df['close'].shift(n)) / df['close'].shift(n)
    
    short_term = 5
    long_term = 20
    df['Short_Term_Momentum'] = (df['close'].rolling(window=short_term).mean() / df['close'].shift(short_term) - 1)
    df['Long_Term_Momentum'] = (df['close'].rolling(window=long_term).mean() / df['close'].shift(long_term) - 1)
    df['Relative_Momentum'] = df['Short_Term_Momentum'] - df['Long_Term_Momentum']
    
    ma_14 = df['close'].rolling(window=14).mean()
    df['MA_14_Signal'] = np.where(df['close'] > ma_14, 1, np.where(df['close'] < ma_14, -1, 0))
    
    def rsi(data, window=14):
        diff = data.diff(1)
        up_chg = 0 * diff
        down_chg = 0 * diff
        up_chg[diff > 0] = diff[diff > 0]
        down_chg[diff < 0] = diff[diff < 0]
        up_chg_avg = up_chg.ewm(com=window-1, min_periods=window).mean()
        down_chg_avg = down_chg.ewm(com=window-1, min_periods=window).mean().abs()
        rs = up_chg_avg / down_chg_avg
        rsi = 100 - 100 / (1 + rs)
        return rsi
