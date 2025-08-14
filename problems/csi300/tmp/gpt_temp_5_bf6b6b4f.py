import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Open-to-Close Move
    open_close_move = df['close'] - df['open']
    
    # Compute Intraday Momentum Direction
    momentum_direction = (high_low_range + open_close_move) / 2
    
    # Adjust for Volume Impact
    volume_adjusted_momentum = momentum_direction * df['volume']
    
    # Apply Time Decay to Historical Signals
    decay_factor = 0.95  # Example decay factor
    factor_values = volume_adjusted_momentum.copy()
    for i in range(1, len(factor_values)):
        factor_values[i] = decay_factor * factor_values[i-1] + (1 - decay_factor) * factor_values[i]
    
    return pd.Series(factor_values, index=df.index, name='intraday_momentum_reversal')
