def heuristics_v2(df):
    # Calculate the daily return
    df['Return'] = df['close'].pct_change()
    
    # Shift the return to align with the factors for prediction
    df['Future_Return'] = df['Return'].shift(-1)
    
    # Drop rows with NaN values resulting from the shift
    df = df.dropna()
    
    # Calculate the 20-day and 100-day simple moving averages of the return
    sma_20_return = df['Return'].rolling(window=20).mean()
    sma_100_return = df['Return'].rolling(window=100).mean()
    
    # Initialize an empty DataFrame to store the dynamic weights
    weights = pd.DataFrame(index=df.index, columns=['sma_20_return', 'sma_100_return'], dtype='float64')
    
    # Compute the dynamic weights
    for col in ['sma_20_return', 'sma_100_return']:
        corr_with_return = df[col].ewm(span=20).corr(df['Future_Return'])
        vol = df[col].ewm(span=20).std()
        weights[col] = (corr_with_return / vol)  # Adjust weights based on volatility and correlation
    
    # Fill any remaining NaNs in the weights matrix (e.g., due to initial periods)
    weights = weights.fillna(0)
    
    # Calculate the heuristic factor
    heuristics_matrix = (pd.concat([sma_20_return, sma_100_return], axis=1) * weights).sum(axis=1)
    
    return heuristics_matrix
