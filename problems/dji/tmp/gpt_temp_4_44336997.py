import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Log Return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate 20-Day Moving Average of Close Price
    df['ma_20'] = df['close'].rolling(window=20).mean()
    
    # Calculate 20-Day Standard Deviation of Log Returns
    df['std_20'] = df['log_return'].rolling(window=20).std()
    
    # Calculate Trend Momentum Indicator
    df['trend_momentum'] = (df['close'] - df['ma_20']) / df['std_20']
    
    # Calculate Intraday High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Volume-Averaged Price
    df['volume_averaged_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4 * df['volume']
    df['volume_averaged_price'] = df['volume_averaged_price'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Calculate True Range
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2].shift(1)), abs(x[1] - x[2].shift(1))), axis=1)
    
    # Combine Trend and Intraday Components
    df['intraday_component'] = df['high_low_spread'] * df['close_to_open_return']
    
    # Volume-Adjusted Momentum
    df['volume_adjusted_momentum'] = df['volume_averaged_price'] / df['true_range']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['trend_momentum'] + df['intraday_component'] + df['volume_adjusted_momentum']
    
    return df['alpha_factor']
