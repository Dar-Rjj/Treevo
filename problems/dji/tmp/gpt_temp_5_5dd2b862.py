def heuristics_v2(df):
    def calculate_weighted_moving_average(column, window, volume):
        return (column.rolling(window=window).mean() * volume.rolling(window=window).mean()).fillna(0)
    
    windows = [5, 10, 20, 60]
    close_prices = df['close']
    volumes = df['volume']
    heuristics_matrix = pd.Series(index=df.index, dtype='float64')
    
    for w in windows:
        heuristics_matrix += calculate_weighted_moving_average(close_prices, w, volumes)
    
    return heuristics_matrix
