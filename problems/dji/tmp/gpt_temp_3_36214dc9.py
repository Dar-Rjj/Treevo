def heuristics_v2(df):
    def sma(price, window):
        return price.rolling(window=window).mean()

    def atr(df, window=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.DataFrame({'tr': [max(high.iloc[i] - low.iloc[i], abs(high.iloc[i] - close.iloc[i-1]), abs(low.iloc[i] - close.iloc[i-1])) for i in range(1, len(df))]}, index=df.index[1:])
        atr = tr.rolling(window=window).mean()
        return atr

    sma_50 = sma(df['close'], 50)
    sma_200 = sma(df['close'], 200)
    macd_sma = sma_50 - sma_200
    atr_factor = atr(df, 14)
    combined_factor = (macd_sma + atr_factor).rename('combined_factor')
    heuristics_matrix = combined_factor.cumsum().rename('heuristic_factor')

    return heuristics_matrix
