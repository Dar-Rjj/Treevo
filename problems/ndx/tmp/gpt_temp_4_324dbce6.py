def heuristics_v2(df):
    # Calculate the rate of change of trading volume
    volume_change = df['volume'].pct_change()
    
    # Calculate the 10-day momentum of closing prices
    close_momentum = df['close'].pct_change(periods=10)
    
    # Combine the two signals
    combined_signal = (volume_change * close_momentum).rolling(window=5).mean()
    
    heuristics_matrix = pd.Series(combined_signal, index=df.index)
    return heuristics_matrix
