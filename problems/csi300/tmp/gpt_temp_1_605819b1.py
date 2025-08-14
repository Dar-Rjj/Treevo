def heuristics_v2(df):
    # Calculate the difference between high and low prices
    intraday_volatility = df['high'] - df['low']
    
    # Calculate the absolute difference between high and open, and low and close
    directional_movement = abs(df['high'] - df['open']) + abs(df['low'] - df['close'])
    
    # Evaluate the momentum by comparing the current close with the previous close
    momentum_1d = df['close'].diff()
    
    # Extend the momentum analysis over the past 5 days
    momentum_5d = df['close'].diff(5)
    
    # Measure the relative position of the closing price within the day's range (high-low)
    relative_position = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Introduce a weighted version where the weights are determined by the volume traded at each price point
    weighted_relative_position = (df['close'] - df['low']) * df['volume'] / (df['high'] - df['low']) * df['volume']
    
    # Investigate the ratio of the close to open prices
    close_open_ratio = df['close'] / df['open']
    
    # Combine the ratio with the percentage change in volume to create a volume-adjusted return
