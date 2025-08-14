def heuristics_v2(df):
    # Calculate Adjusted High-Low Spread
    df['High_Low_Spread'] = df['high'] - df['low']
    
    # Weight by Volume
    df['Volume_Weighted_High_Low_Spread'] = df['High_Low_Spread'] * df['volume']
    
    # Apply Conditional Weight to High-Low Spread
    df['Close_to_Open_Return'] = (df['close'] - df['open']) / df['open']
    positive_return_weight = 1.5
