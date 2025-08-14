def heuristics_v2(df):
    def calculate_ema(column, span=10):
        return column.ewm(span=span, adjust=False).mean()
    
    def calculate_pvi_nvi(close, volume):
        pvi = [1]
        nvi = [1]
        for i in range(1, len(volume)):
            if volume[i] > volume[i-1]:
                pvi.append(pvi[-1] * (close[i] / close[i-1]))
                nvi.append(nvi[-1])
            elif volume[i] < volume[i-1]:
                nvi.append(nvi[-1] * (close[i] / close[i-1]))
                pvi.append(pvi[-1])
            else:
                pvi.append(pvi[-1])
                nvi.append(nvi[-1])
        return pvi, nvi

    ema_close = calculate_ema(df['close'])
    pvi, nvi = calculate_pvi_nvi(df['close'], df['volume'])
    pvi = pd.Series(pvi, index=df.index)
    nvi = pd.Series(nvi, index=df.index)
    
    heuristics_matrix = (ema_close + pvi - nvi) / 3
    return heuristics_matrix
