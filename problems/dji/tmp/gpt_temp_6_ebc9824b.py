def heuristics_v2(df):
    def sma(price, window):
        return price.rolling(window=window).mean()

    df['SMA_50'] = sma(df['close'], 50)
    df['SMA_200'] = sma(df['close'], 200)
    df['SMA_Ratio'] = df['SMA_50'] / df['SMA_200']
    df['Daily_Range'] = df['high'] - df['low']
    df['Adjusted_Volume'] = df['volume'].apply(lambda x: np.log(x + 1))  # Adding 1 to avoid log(0)
    df['Heuristic_Factor'] = (df['SMA_Ratio'] * df['Daily_Range']) * df['Adjusted_Volume']
    
    heuristics_matrix = df['Heuristic_Factor'].dropna()
    return heuristics_matrix
