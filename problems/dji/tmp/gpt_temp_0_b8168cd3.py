import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, ema_period=10):
    # Calculate Weighted Close Price
    df['weighted_close'] = df['close'] * df['volume']
    
    # Calculate Average True Range
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x['high'], x['prev_close']) - min(x['low'], x['prev_close']), axis=1)
    df['atr'] = df['true_range'].rolling(window=ema_period).mean()
    
    # Compute Exponential Moving Average of Weighted Close Price
    df['ema_weighted_close'] = df['weighted_close'].ewm(span=ema_period, adjust=False).mean()
    
    # Compute Momentum from Weighted Close Price
    df['momentum'] = df['weighted_close'].pct_change(periods=ema_period)
    
    # Adjust Momentum
    df['volume_change'] = df['volume'].pct_change()
    df['adjusted_momentum'] = df['momentum'] / (df['atr'] + 1e-6) * (1 + df['volume_change'])
    
    return df['adjusted_momentum']

# Example usage:
# df = pd.DataFrame({
#     'open': [100, 105, 103, 107, 108],
#     'high': [105, 108, 106, 110, 112],
#     'low': [98, 103, 101, 105, 106],
#     'close': [104, 107, 105, 110, 111],
#     'volume': [1000, 1200, 1100, 1300, 1400]
# })
# print(heuristics_v2(df))
