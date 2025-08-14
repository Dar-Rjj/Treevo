def heuristics_v2(df):
    # Calculate the 10-day rate of change (ROC) for the closing price
    df['ROC10'] = df['close'].pct_change(periods=10)
    
    # Calculate the 21-day Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=21).mean()
    df['BB_std'] = df['close'].rolling(window=21).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # Adjust the Bollinger Band width by the 21-day average volume
    df['SMA21_vol'] = df['volume'].rolling(window=21).mean()
    df['BB_width_adjusted'] = df['BB_width'] / df['SMA21_vol']
    
    # Compile heuristics into a matrix
    heuristics_matrix = pd.Series(index=df.index, dtype='float64')
    heuristics_matrix = df['ROC10'] * df['BB_width_adjusted']
    
    return heuristics_matrix
