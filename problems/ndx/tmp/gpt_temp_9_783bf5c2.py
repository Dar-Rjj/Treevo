def heuristics_v2(df):
    # Calculate price change from open to close
    price_change_oc = df['close'] - df['open']
    
    # Calculate price change from high to low
    price_change_hl = df['high'] - df['low']
    
    # Calculate the ratio of close to open
    close_open_ratio = df['close'] / df['open']
    
    # Calculate the ratio of high to low
    high_low_ratio = df['high'] / df['low']
    
    # Calculate the percentage change in volume
    volume_change_pct = df['volume'].pct_change()
    
    # Create a DataFrame for the heuristics
    heuristics_matrix = pd.DataFrame({
        'price_change_oc': price_change_oc,
        'price_change_hl': price_change_hl,
        'close_open_ratio': close_open_ratio,
        'high_low_ratio': high_low_ratio,
        'volume_change_pct': volume_change_pct
    })
    
    # Return the heuristics as a Series
    return heuristics_matrix
