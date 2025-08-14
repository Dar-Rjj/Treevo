import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=20, m=14, k=14):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Volume Change Ratio
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Weighted Momentum
    df['weighted_momentum'] = (df['daily_return'] * df['volume_ratio']).rolling(window=n).sum()
    
    # True Range
    df['true_range'] = df.apply(lambda row: max(row['high'] - row['low'], 
                                                 abs(row['high'] - row['close'].shift(1)), 
                                                 abs(row['low'] - row['close'].shift(1))), axis=1)
    
    # ATR
    df['atr'] = df['true_range'].rolling(window=m).mean()
    
    # Adjust for Enhanced Volatility
    df['enhanced_atr'] = df['atr'] * (1 + 0.5 * (df['high'] - df['low']) / df['close'].shift(1))
    df['adjusted_momentum'] = df['weighted_momentum'] - df['enhanced_atr']
    
    # Calculate +DM and -DM
    df['+dm'] = (df['high'] - df['high'].shift(1)).apply(lambda x: max(x, 0))
    df['-dm'] = (df['low'].shift(1) - df['low']).apply(lambda x: max(x, 0))
    
    # Smooth +DM and -DM over k periods
    df['smoothed_+dm'] = df['+dm'].ewm(span=k, min_periods=k).mean()
    df['smoothed_-dm'] = df['-dm'].ewm(span=k, min_periods=k).mean()
    
    # +DI and -DI
    df['+di'] = 100 * (df['smoothed_+dm'] / df['atr'])
    df['-di'] = 100 * (df['smoothed_-dm'] / df['atr'])
    
    # ADX
    df['adx'] = 100 * (abs(df['+di'] - df['-di']) / (df['+di'] + df['-di']))
    
    # Final Factor
    df['final_factor'] = df['adjusted_momentum'] * (df['adx'] / 100)
    
    return df['final_factor']
