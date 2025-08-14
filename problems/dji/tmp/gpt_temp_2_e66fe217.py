def heuristics_v2(df):
    # Calculate 5-day logarithmic return for close
    df['Log_Return'] = np.log(df['close']).diff(5)
    
    # Calculate the ratio of the highest to the lowest high price over the past 20 days
    df['High_Low_Ratio'] = df['high'].rolling(window=20).max() / df['high'].rolling(window=20).min()
    
    # Calculate 10-day percentage change for volume
    df['Volume_Percent_Change'] = df['volume'].pct_change(periods=10)
    
    # Generate the heuristics factor
    df['Heuristic_Factor'] = (df['Log_Return'] * df['High_Low_Ratio']) + df['Volume_Percent_Change']
    
    heuristics_matrix = df['Heuristic_Factor'].dropna()
    return heuristics_matrix
