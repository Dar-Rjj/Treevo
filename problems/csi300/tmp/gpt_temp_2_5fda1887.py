import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Weight by Volume
    df['volume_weighted_close_to_open_return'] = df['close_to_open_return'] * df['volume']

    # Determine Dynamic Window Size Based on Volume
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['window_size'] = (df['volume'] / df['volume_ma']).apply(lambda x: max(int(10 * x), 5))
    
    # Calculate Adaptive Price Momentum
    momentum = [None] * len(df)
    for i in range(len(df)):
        if i < df['window_size'][i]:
            continue
        momentum[i] = (df.iloc[i]['close'] - df.iloc[i - df['window_size'][i]]['close']) / df.iloc[i - df['window_size'][i]]['close']
    df['adaptive_price_momentum'] = momentum
    
    # Calculate Intraday High-Low Volatility
    df['high_low_volatility'] = (df['high'] - df['low']) / df['close']
    
    # Combine Factors
    df['factor_value'] = (df['volume_weighted_close_to_open_return'] + 
                           df['adaptive_price_momentum'] + 
                           df['high_low_volatility']) / 3
    
    return df['factor_value']
