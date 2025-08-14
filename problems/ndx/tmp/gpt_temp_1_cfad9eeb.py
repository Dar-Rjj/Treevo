import pandas as pd

def heuristics_v2(df):
    def dmi(df, n=14):
        df['up_move'] = df['high'].diff()
        df['down_move'] = -df['low'].diff()
        df['+dm'] = 0
        df['-dm'] = 0
        df.loc[(df['up_move'] > df['down_move']) & (df['up_move'] > 0), '+dm'] = df['up_move']
        df.loc[(df['down_move'] > df['up_move']) & (df['down_move'] > 0), '-dm'] = df['down_move']
        
        atr = df[['high', 'low', 'close']].rolling(window=n).apply(lambda x: (max(x)-min(x)).mean(), raw=True)
        di_pos = 100 * (df['+dm'].rolling(window=n).sum() / atr).replace([np.inf, -np.inf], 0).fillna(0)
        di_neg = 100 * (df['-dm'].rolling(window=n).sum() / atr).replace([np.inf, -np.inf], 0).fillna(0)
        
        return di_pos - di_neg

    def mfi(df, n=14):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        pos_money_flow = (typical_price > typical_price.shift(1)) * money_flow
        neg_money_flow = (typical_price < typical_price.shift(1)) * money_flow
        
        mfr = pos_money_flow.rolling(window=n).sum() / neg_money_flow.rolling(window=n).sum()
        mfi_series = 100 - (100 / (1 + mfr))
        
        return mfi_series.replace([np.inf, -np.inf], 0).fillna(0)

    def sma(df, n=50):
        return df['close'].rolling(window=n).mean()

    df['dmi'] = dmi(df)
    df['mfi'] = mfi(df)
    df['sma'] = sma(df)
    
    heuristics_matrix = df[['dmi', 'mfi', 'sma']].mean(axis=1).rename('heuristics')
    
    return heuristics_matrix
