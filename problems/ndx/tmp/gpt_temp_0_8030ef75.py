import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=20, m=14):
    # Calculate Daily Log Return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate Volume Change Ratio
    df['avg_volume_n_days'] = df['volume'].rolling(window=n).mean()
    df['volume_change_ratio'] = df['volume'] / df['avg_volume_n_days']
    
    # Compute Weighted Momentum
    df['weighted_momentum'] = (df['log_return'] * df['volume_change_ratio']).rolling(window=n).sum()
    
    # Adjust for Price Volatility
    df['true_range'] = df.apply(lambda row: max(row['high'] - row['low'], 
                                                abs(row['high'] - row['close'].shift(1)), 
                                                abs(row['low'] - row['close'].shift(1))), axis=1)
    df['avg_true_range'] = df['true_range'].rolling(window=m).mean()
    
    df['adjusted_weighted_momentum'] = df['weighted_momentum'] - df['avg_true_range']
    
    return df['adjusted_weighted_momentum']
