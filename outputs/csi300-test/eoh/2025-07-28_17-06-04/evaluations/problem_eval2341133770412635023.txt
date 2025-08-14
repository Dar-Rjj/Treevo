def heuristics_v2(df):
    def calc_factor(row, avg_close, avg_volume):
        return (avg_close - row['open']) * (row['volume'] / avg_volume)
    
    avg_close = df['close'].rolling(window=5).mean()
    avg_volume = df['volume'].rolling(window=5).mean()
    
    heuristics_matrix = df.apply(calc_factor, args=(avg_close, avg_volume), axis=1)
    
    return heuristics_matrix
