def heuristics_v2(df):
    # Calculate the daily return
    df['Return'] = df['close'].pct_change()
    
    # Shift the return to align with the factors for prediction
    df['Future_Return'] = df['Return'].shift(-1)
    
    # Drop rows with NaN values resulting from the shift
    df = df.dropna()
    
    # Initialize an empty DataFrame to store the dynamic weights
    weights = pd.DataFrame(index=df.index, columns=df.columns[:-2], dtype='float64')
    
    # Compute the dynamic weights
    for col in df.columns[:-2]:
        corr_with_return = df[col].ewm(span=20).corr(df['Future_Return'])
        vol = df[col].ewm(span=20).std()
        abs_corr = df[col].ewm(span=20).corr(df['Future_Return']).abs()
        weights[col] = (corr_with_return / (vol * abs_corr))  # Adjust weights based on volatility, correlation, and absolute correlation
    
    # Fill any remaining NaNs in the weights matrix (e.g., due to initial periods)
    weights = weights.fillna(0)
    
    # Calculate the heuristic factor
    heuristics_matrix = (df[df.columns[:-2]] * weights).sum(axis=1)
    
    return heuristics_matrix
