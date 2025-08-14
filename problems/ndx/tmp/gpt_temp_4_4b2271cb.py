def heuristics_v2(df):
    # Calculate the 20-day moving average of daily high and low prices
    avg_high_low = (df['high'] + df['low']) / 2
    moving_avg_20_high_low = avg_high_low.rolling(window=20).mean()
    
    # Calculate the 50-day moving average of daily closing prices
    moving_avg_50_close = df['close'].rolling(window=50).mean()
    
    # Compute the heuristic factor as the difference between the two moving averages
    heuristic_factor = moving_avg_20_high_low - moving_avg_50_close
    
    # Create a Series for the heuristic factor, ensuring it is indexed correctly
    heuristics_matrix = pd.Series(heuristic_factor, index=df.index)
    return heuristics_matrix
