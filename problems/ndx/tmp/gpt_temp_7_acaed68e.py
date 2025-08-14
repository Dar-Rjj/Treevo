def heuristics_v2(df):
    # Calculate 10-day and 50-day Simple Moving Averages
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Calculate the ratio between the 10-day and 50-day SMAs
    df['SMA_Ratio'] = df['SMA_10'] / df['SMA_50']
    
    # Calculate daily log returns
    df['Log_Return'] = np.log(df['close']).diff()
    
    # Calculate 20-day standard deviation of daily log returns as a measure of volatility
    df['Volatility'] = df['Log_Return'].rolling(window=20).std()
    
    # Compute the heuristic score as a composite of momentum (SMA ratio) and volatility
    df['Heuristic_Score'] = df['SMA_Ratio'] + df['Volatility']
    
    # The output is a Series indexed by (date)
    heuristics_matrix = df['Heuristic_Score'].copy()
    
    return heuristics_matrix
