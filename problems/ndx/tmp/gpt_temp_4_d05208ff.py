def heuristics_v2(df):
    # Calculate the 50-day and 200-day simple moving averages of the close price
    sma_50 = df['close'].rolling(window=50).mean()
    sma_200 = df['close'].rolling(window=200).mean()
    
    # Calculate the ratio between the SMAs
    sma_ratio = sma_50 / sma_200
    
    # Calculate the daily log return
    df['Log_Return'] = np.log(df['close']).diff()
    
    # Calculate the 60-day mean absolute deviation of daily log returns
    mad_60 = df['Log_Return'].rolling(window=60).apply(lambda x: (x - x.mean()).abs().mean(), raw=False)
    
    # Generate the heuristic matrix by dividing the SMA ratio with the MAD
    heuristics_matrix = sma_ratio / mad_60
    
    return heuristics_matrix
