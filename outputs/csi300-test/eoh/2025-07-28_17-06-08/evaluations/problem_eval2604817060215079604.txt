import pandas as pd

def heuristics_v2(df):
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def adx(df, period=14):
        df['TR'] = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
        df['+DM'] = (df['high'].diff().where(lambda x: x > 0, 0)).combine(df['high'] - df['high'].shift(), max)
        df['-DM'] = (df['low'].diff().where(lambda x: x < 0, 0)).combine(df['low'].shift() - df['low'], max)
        df['+DI'] = 100 * (df['+DM'] / df['TR']).rolling(window=period).mean()
        df['-DI'] = 100 * (df['-DM'] / df['TR']).rolling(window=period).mean()
        df['ADX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])).rolling(window=period).mean()
        return df['ADX']
    
    def mfi(df, period=14):
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        df['mf_positive'] = mf.where(df['close'] > df['close'].shift(1), 0)
        df['mf_negative'] = mf.where(df['close'] < df['close'].shift(1), 0)
        mfr = (df['mf_positive'].rolling(window=period).sum() / df['mf_negative'].rolling(window=period).sum())
        return 100 - (100 / (1 + mfr))

    corr_rsi_return = 0.5  # Example value
    corr_adx_return = 0.3   # Example value
    corr_mfi_return = 0.2   # Example value
    
    heuristics_matrix = (corr_rsi_return * rsi(df['close']) + 
                         corr_adx_return * adx(df) + 
                         corr_mfi_return * mfi(df))
    return heuristics_matrix
