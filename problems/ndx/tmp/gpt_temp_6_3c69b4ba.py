def heuristics_v2(df):
    # Momentum factor (1-month return)
    momentum = df['close'].pct_change(periods=21)
    
    # Volatility factor (standard deviation of daily returns over 1 month)
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=21).std()
    
    # Liquidity factor (volume relative to its 1-month average)
    avg_volume = df['volume'].rolling(window=21).mean()
    liquidity = df['volume'] / avg_volume
    
    # Combine factors into a DataFrame
    heuristics_matrix = pd.concat([momentum, volatility, liquidity], axis=1)
    heuristics_matrix.columns = ['Momentum', 'Volatility', 'Liquidity']
    
    return heuristics_matrix
