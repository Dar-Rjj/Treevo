def heuristics_v2(df):
    # Calculate the 20-day and 60-day simple moving averages of the volume
    sma_vol_20 = df['volume'].rolling(window=20).mean()
    sma_vol_60 = df['volume'].rolling(window=60).mean()
    
    # Calculate the ratio between the 20-day and 60-day SMAs of volume
    vol_ratio = sma_vol_20 / sma_vol_60
    
    # Calculate the daily log return
    df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate the 40-day rolling skewness of daily log returns
    skew_40 = df['Log_Return'].rolling(window=40).apply(lambda x: pd.Series(x).skew(), raw=False)
    
    # Generate the heuristic matrix by multiplying the volume ratio with the skewness
    heuristics_matrix = vol_ratio * skew_40
    
    return heuristics_matrix
