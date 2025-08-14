def heuristics_v2(df):
    """
    Calculate the Volume Weighted Close-to-Open Return.
    
    Parameters:
    - df: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    - pandas Series with the Volume Weighted Close-to-Open Return for each date.
    """
    # Calculate the Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) / df['open']
    
    # Weight by Volume
    volume_weighted_return = close_to_open_return * df['volume']
    
    return volume_weighted_return
