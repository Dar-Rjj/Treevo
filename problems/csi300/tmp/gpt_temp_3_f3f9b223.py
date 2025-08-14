import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Compute Average Volume (last 20 days)
    average_volume = df['volume'].rolling(window=20).mean()
    
    # Determine Weights based on recent volume trend
    weights = [0.7, 0.3] if df['volume'] > average_volume else [0.5, 0.5]
    
    # Weighted Combination of Intraday Range and Close-to-Open Return
    weighted_combination = (weights[0] * intraday_range + 
                            weights[1] * close_to_open_return)
    
    # Consider Momentum
    momentum = (df['close'] - df['open']) / df['open']
    
    # Add Momentum to the Weighted Combination
    weighted_combination += momentum
    
    # Consider Reversal
    reversal = (df['open'] - df['close']) / df['open']
    
    # Subtract Reversal from the Weighted Combination
    factor_values = weighted_combination - reversal
    
    return factor_values
