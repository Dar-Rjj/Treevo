import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Intraday High-Low Spread
    df['high_low_spread'] = (df['high'] - df['low']) * df['volume']
    
    # Combine Intraday High-Low Spread and Intraday Return
    df['combined_intraday_factor'] = df['high_low_spread'] * df['intraday_return']
    
    # Incorporate Volume Influence
    N = 10  # Example: 10-day moving average
    df['avg_volume'] = df['volume'].rolling(window=N).mean()
    df['volume_impact'] = df['volume'] / df['avg_volume']
    df['weighted_intraday_factor'] = df['combined_intraday_factor'] * df['volume_impact']
    
    # Calculate True Range for each day
    df['true_range'] = df.apply(lambda row: max(row['high'] - row['low'], 
                                                abs(row['high'] - df['close'].shift(1)), 
                                                abs(row['low'] - df['close'].shift(1))), axis=1)
    
    # Calculate 14-day Simple Moving Average of the True Range
    df['sma_true_range_14'] = df['true_range'].rolling(window=14).mean()
    
    # Construct the Momentum Component
    df['momentum_component'] = (df['close'] - df['sma_true_range_14']) / df['sma_true_range_14']
    
    # Enhance with Volume-Weighted High-Low Difference
    df['high_low_diff'] = df['high'] - df['low']
    df['volume_weighted_high_low_diff'] = df['high_low_diff'] * df['volume']
    
    # Calculate Daily Log Returns
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
    
    # Compute Realized Volatility
    volatility_window = 20
    df['realized_volatility'] = df['log_return'].rolling(window=volatility_window).std()
    
    # Normalize Momentum by Volatility
    df['normalized_momentum'] = df['momentum_component'] / df['realized_volatility']
    
    # Introduce Trend Component
    df['sma_close_50'] = df['close'].rolling(window=50).mean()
    df['trend_direction'] = (df['close'] > df['sma_close_50']).astype(int) * 2 - 1
    
    # Synthesize Alpha Factor
    df['alpha_factor'] = (df['normalized_momentum'] + 
                          df['weighted_intraday_factor'] + 
                          df['intraday_return'].rolling(window=14).mean() * df['volume'] + 
                          df['volume_weighted_high_low_diff']) * df['trend_direction']
    
    return df['alpha_factor']
