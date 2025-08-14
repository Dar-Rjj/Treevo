import numpy as np
def heuristics(df):
    # Calculate Intraday High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']
    
    # Compute Intraday Range Weighted Average Price (IRWAP)
    df['irwap'] = (df['close'] * df['volume']).rolling(window=1).sum() / df['volume'].rolling(window=1).sum()
    
    # Evaluate IRWAP Difference
    df['irwap_diff'] = df['close'] - df['irwap']
    
    # Calculate Adjusted Volume
    df['adjusted_volume'] = df['volume'] / df['high_low_spread']
    
    # Integrate Volume-Adjusted Log Return
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
