import pandas as pd

def heuristics_v2(df):
    def rsi(series, periods=14):
        delta = series.diff()
        up, down = delta.clip(lower=0), delta.clip(upper=0).abs()
        avg_gain = up.rolling(window=periods, min_periods=1).mean()
        avg_loss = down.rolling(window=periods, min_periods=1).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def pvi(nv, pv, ncp, pop):
        if ncp > pop:
            return nv + pv * (ncp / pop)
        else:
            return nv

    def nvi(nv, pv, ncp, pop):
        if ncp < pop:
            return nv + pv * (pop / ncp)
        else:
            return nv
    
    df['RSI'] = rsi(df['close'])
    
    df['PVI'] = [pvi(df['volume'].iloc[0], 1, df['close'].iloc[0], df['open'].iloc[0])] + [pvi(df['volume'].iloc[i], df['PVI'].iloc[i-1], df['close'].iloc[i], df['open'].iloc[i]) for i in range(1, len(df))]
    df['NVI'] = [nvi(df['volume'].iloc[0], 1, df['close'].iloc[0], df['open'].iloc[0])] + [nvi(df['volume'].iloc[i], df['NVI'].iloc[i-1], df['close'].iloc[i], df['open'].iloc[i]) for i in range(1, len(df))]
    
    df['Strength_Ratio'] = df['high'].rolling(window=20).max() / df['low'].rolling(window=20).min()
    
    heuristics_matrix = ((df['RSI'] + df['PVI'] - df['NVI']) * df['Strength_Ratio']).rename('heuristic_factor')
    
    return heuristics_matrix
