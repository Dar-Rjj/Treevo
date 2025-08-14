def heuristics_v2(df):
    # Calculate momentum (close price change)
    momentum = df['close'].pct_change()
    
    # Calculate volatility (standard deviation of daily returns over a rolling window)
    volatility = df['close'].pct_change().rolling(window=30).std()
    
    # Calculate average volume over a rolling window
    avg_volume = df['volume'].rolling(window=30).mean()
    
    # Compute the dynamic weight for each factor
    weight_momentum = abs(momentum).rolling(window=60).corr(df['close'].pct_change().shift(-1)).fillna(0)
    weight_volatility = abs(volatility).rolling(window=60).corr(df['close'].pct_change().shift(-1)).fillna(0)
    weight_volume = abs(avg_volume).rolling(window=60).corr(df['close'].pct_change().shift(-1)).fillna(0)
    
    # Normalize the weights
    total_weight = weight_momentum + weight_volatility + weight_volume
    weight_momentum /= total_weight
    weight_volatility /= total_weight
    weight_volume /= total_weight
    
    # Calculate the heuristics factor
    heuristics_factor = (momentum * weight_momentum) + (volatility * weight_volatility) + (avg_volume * weight_volume)
    
    return heuristics_matrix
