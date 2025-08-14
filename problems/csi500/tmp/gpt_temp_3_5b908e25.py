import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Compute Volume-Weighted High-Low Spread
    volume_weighted_high_low_spread = (df['high'] - df['low']) * df['volume']
    
    # Compute Logarithmic Return
    log_return = np.log(df['close'] / df['close'].shift(1))
    
    # Compute Refined Momentum Calculation with Weighted Average by Volume
    recent_log_returns = log_return.rolling(window=5).apply(lambda x: np.average(x, weights=df.loc[x.index, 'volume']), raw=False)
    
    # Compute Gap Difference
    gap_difference = df['open'] - df['close'].shift(1)
    
    # Adjust for Volume on Gap Day
    scaled_gap_difference = gap_difference * df['volume']
    
    # Combine Intraday Volatility and Momentum
    combined_factor = high_low_spread + volume_weighted_high_low_spread + recent_log_returns
    
    # Final Alpha Factor
    final_alpha_factor = combined_factor + scaled_gap_difference
    
    return final_alpha_factor
