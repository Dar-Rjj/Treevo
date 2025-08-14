import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Gap
    df['price_gap'] = df['open'] - df['close'].shift(1)
    
    # Calculate Momentum for N = 5 and N = 10
    for N in [5, 10]:
        # Exponential Moving Average (EMA)
        df[f'ema_{N}'] = df['close'].ewm(span=N, adjust=False).mean()
        
        # Price Difference
        df[f'price_diff_{N}'] = df['close'] - df[f'ema_{N}']
        
        # Momentum Score
        df[f'momentum_score_{N}'] = df[f'price_diff_{N}'] / df[f'ema_{N}']
    
    # Weight by Volume
    for N in [5, 10]:
        df[f'volume_adjusted_momentum_{N}'] = (df[f'momentum_score_{N}'] / df['volume']) * df['price_gap']
    
    # Adjust Momentum Score by Volume-Weighted Factor
    for N in [5, 10]:
        # Daily Returns
        df['daily_return'] = df['high'] - df['close'].shift(1)
        
        # Aggregate Product of Daily Returns and Volume over N days
        df[f'aggregate_product_{N}'] = (df['daily_return'] * df['volume']).rolling(window=N).sum()
        
        # Aggregate Volume over N days
        df[f'aggregate_volume_{N}'] = df['volume'].rolling(window=N).sum()
        
        # Final Volume-Weighted Factor
        df[f'final_volume_weighted_factor_{N}'] = df[f'aggregate_product_{N}'] / df[f'aggregate_volume_{N}']
        
        # Integrate
        df[f'integrated_momentum_{N}'] = df[f'volume_adjusted_momentum_{N}'] * df[f'final_volume_weighted_factor_{N}']
    
    # Output: Integrated Price and Volume Adjusted Momentum
    return df[['integrated_momentum_5', 'integrated_momentum_10']]
