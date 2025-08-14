import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=20, alpha=0.1):
    # Calculate Adjusted Close Price
    adjusted_close = df['close'] - df['open']
    
    # Enhanced Intraday Momentum
    high_low_spread = df['high'] - df['low']
    intraday_momentum = (high_low_spread - df['open']).abs()
    
    # Volume Impact on Intraday Momentum
    smoothed_volume = df['volume'].ewm(alpha=alpha).mean()
    volume_impacted_intraday = intraday_momentum * smoothed_volume
    
    # Long-Term Momentum Adjusted for Volume
    long_term_momentum = (df['close'].shift(-n) - df['close']) / df['close'].shift(-n)
    volume_ratio = df['volume'] / df['volume'].shift(-n)
    volume_adjusted_long_term = long_term_momentum * volume_ratio
    
    # Consolidate All Factors
    total_factor = adjusted_close + volume_impacted_intraday + volume_adjusted_long_term
    
    # Calculate Price Volatility
    true_range = np.maximum.reduce([df['high'] - df['low'],
                                    (df['high'] - df['close'].shift(1)).abs(),
                                    (df['low'] - df['close'].shift(1)).abs()])
    average_true_range = true_range.rolling(window=n).mean()
    
    # Adjust the Total Factor for Price Volatility
    final_factor = (total_factor / average_true_range) * df['open']
    
    return final_factor
