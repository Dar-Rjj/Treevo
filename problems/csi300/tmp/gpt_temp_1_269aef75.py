def heuristics_v2(df):
    # Calculate 20-Day Average Close (t-19, t-18, ..., t)
    df['20_day_avg_close'] = df['close'].rolling(window=20).mean()
    
    # Subtract from Today's Close (t) for Long-Term Momentum
    df['long_term_momentum'] = df['close'] - df['20_day_avg_close']
    
    # Compute 5-Day Moving Average of Close Price (t-4, t-3, ..., t)
    df['5_day_avg_close'] = df['close'].rolling(window=5).mean()
    
    # Subtract 5-Day Moving Average from Current Close Price (t)
    df['5_day_diff_close'] = df['close'] - df['5_day_avg_close']
    
    # Divide by Maximum 5-Day Close Price Range for Short-Term Momentum
    df['short_term_momentum'] = df['5_day_diff_close'] / df['close'].rolling(window=5).max()
    
    # Calculate 10-Day Moving Average of Close Price (t-9, t-8, ..., t)
    df['10_day_avg_close'] = df['close'].rolling(window=10).mean()
    
    # Subtract 10-Day Moving Average from Current Close Price (t)
    df['10_day_diff_close'] = df['close'] - df['10_day_avg_close']
    
    # Divide by Maximum 10-Day Close Price Range for Medium-Term Momentum
    df['medium_term_momentum'] = df['10_day_diff_close'] / df['close'].rolling(window=10).max()
    
    # Add 5-Day and 10-Day Momentum Indicators
    df['momentum_sum'] = df['short_term_momentum'] + df['medium_term_momentum']
    
    # Divide by 2 for Balanced Momentum Signal
    df['balanced_momentum'] = df['momentum_sum'] / 2
    
    # Subtract 10-day Price Return from 5-day Price Return for Trend Continuation
    df['price_return_5d'] = df['close'].pct_change(5)
