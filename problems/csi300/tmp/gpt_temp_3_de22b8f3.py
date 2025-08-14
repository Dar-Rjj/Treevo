def heuristics_v2(df, lookback=20):
    # Calculate Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Calculate Volume-to-Price Ratio
    volume_to_price_ratio = df['volume'] / ((df['high'] + df['low']) / 2)
    
    # Calculate Weighted Intraday Volatility
    weighted_intraday_volatility = intraday_volatility * volume_to_price_ratio
    
    # Calculate Close-to-Open Change
    close_to_open_change = df['close'] - df['open']
    
    # Calculate Enhanced Factor
    enhanced_factor = weighted_intraday_volatility - close_to_open_change
    
    # Calculate Close-Open Spread Momentum
    close_open_spread_momentum = close_to_open_change.rolling(window=lookback).mean()
    
    # Determine the final factor based on the condition
    final_factor = enhanced_factor.copy()
    for date in df.index:
        if close_open_spread_momentum.loc[date] > 0:
            final_factor.loc[date] = enhanced_factor.loc[date] * (1 + close_open_spread_momentum.loc[date])
        else:
            final_factor.loc[date] = enhanced_factor.loc[date] * (1 - close_open_spread_momentum.loc[date])
    
    return final_factor
