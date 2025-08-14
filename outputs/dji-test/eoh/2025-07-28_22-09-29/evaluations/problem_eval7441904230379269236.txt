def heuristics_v2(df):
    # Calculate 21-day logarithmic return for momentum
    df['Log_Return'] = np.log(df['close']).diff(21)
    
    # Calculate 10-day Average True Range (ATR) for volatility
    df['TR'] = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
    df['ATR_10'] = df['TR'].rolling(window=10).mean()
    
    # Calculate 30-day simple moving average of close price
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    
    # Generate the heuristics factor
    df['Heuristic_Factor'] = df['Log_Return'] + (df['ATR_10'] / df['SMA_30'])
    
    heuristics_matrix = df['Heuristic_Factor'].dropna()
    return heuristics_matrix
