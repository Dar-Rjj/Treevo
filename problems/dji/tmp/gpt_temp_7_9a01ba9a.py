def heuristics_v2(df):
    # Calculate 50-day simple moving average for return (momentum)
    df['Momentum'] = df['close'].pct_change().rolling(window=50).mean()
    
    # Calculate 14-day Average True Range for market volatility
    df['True_Range'] = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
    df['Average_True_Range'] = df['True_Range'].rolling(window=14).mean()
    
    # Calculate 20-day simple moving average for volume as a measure of liquidity
    df['Volume_avg'] = df['volume'].rolling(window=20).mean()
    
    # Generate the heuristics factor
    df['Heuristic_Factor'] = df['Momentum'] * (df['Average_True_Range'] / df['Volume_avg'])
    
    heuristics_matrix = df['Heuristic_Fctor'].dropna()
    return heuristics_matrix
