def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close to Midpoint Deviation
    df['close_to_midpoint_deviation'] = df['close'] - (df['high'] + df['low']) / 2
    
    # Calculate Adjusted Intraday Reversal with Momentum Adjustment
    df['intraday_reversal'] = 2 * (df['high'] - df['low']) / (df['close'] + df['open'])
    df['momentum_adjustment'] = 1 + (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['adjusted_intraday_reversal'] = df['intraday_reversal'] * df['momentum_adjustment']
    
    # Generate Intermediate Alpha Factor 1
    df['previous_day_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
