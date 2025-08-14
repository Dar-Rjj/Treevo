def heuristics_v2(df):
    # Calculate the price range for each day
    df['range'] = df['high'] - df['low']
    
    # Calculate the relative position of the close price within the daily range
    df['close_position'] = (df['close'] - df['low']) / df['range']
    
    # Calculate the average true range over a 14-day period
    df['true_range'] = df[['high' - 'low', abs('high' - df['close'].shift(1)), abs('low' - df['close'].shift(1))]].max(axis=1)
    df['avg_true_range'] = df['true_range'].rolling(window=14).mean()
    
    # Calculate the ratio of the daily range to the 14-day average true range
    df['range_ratio'] = df['range'] / df['avg_true_range']
    
    # Calculate the final alpha factor
    df['alpha_factor'] = (df['close_position'] * df['volume']) / df['range_ratio']
    
    return df['alpha_factor']
