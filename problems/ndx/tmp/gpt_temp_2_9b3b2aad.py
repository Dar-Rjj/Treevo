def heuristics_v2(df):
    # Calculate 21-day and 63-day Simple Moving Averages
    df['SMA_21'] = df['close'].rolling(window=21).mean()
    df['SMA_63'] = df['close'].rolling(window=63).mean()
    
    # Calculate the difference between the 21-day and 63-day SMAs
    df['SMA_Diff'] = df['SMA_21'] - df['SMA_63']
    
    # Calculate daily log returns
    df['Log_Return'] = np.log(df['close']).diff()
    
    # Calculate 30-day standard deviation of daily log returns as a measure of volatility
    df['Volatility'] = df['Log_Return'].rolling(window=30).std()
    
    # Compute the heuristic score as a composite of momentum (SMA difference) and volatility
    df['Heuristic_Score'] = df['SMA_Diff'] + df['Volatility']
    
    # The output is a Series indexed by (date)
    heuristics_matrix = df['Heuristic_Score'].copy()
    
    return heuristics_matrix
