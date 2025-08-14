def heuristics_v2(df):
    # Calculate the standard deviation over a 10-day period as a proxy for volatility
    volatility = df['close'].rolling(window=10).std()
    
    # Compute the relative strength compared to a benchmark (e.g., S&P 500), assumed to be available in the 'benchmark' column
    relative_strength = (df['close'] / df['benchmark']) - 1
    
    # Combine the factors into a single heuristic score
    heuristics_matrix = (volatility > 0) * 1 + (relative_strength > 0) * 1

    return heuristics_matrix
