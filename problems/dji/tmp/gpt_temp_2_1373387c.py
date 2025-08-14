import numpy as np
def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Measure Close Position in Range
    df['close_position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['close_position_in_range'].fillna(0, inplace=True)  # Handle division by zero
    
    # Calculate Volume-Weighted Average Price (VWAP)
    prices = (df[['open', 'high', 'low', 'close']].values * df[['volume']].values).sum(axis=1)
    total_volume = df['volume'].sum()
    df['vwap'] = prices / total_volume
    
    # Calculate Intraday Momentum
    df['high_low_diff'] = df['high'] - df['low']
    df['open_close_return'] = df['close'] / df['open'] - 1
