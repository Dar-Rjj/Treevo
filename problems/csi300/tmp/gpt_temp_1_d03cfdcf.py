import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Expansion Momentum Persistence
    # Calculate Volume-Expanded Returns
    N = 5  # Price return window
    M = 20  # Volume average window
    
    # Compute raw N-day price returns
    price_returns = df['close'].pct_change(periods=N)
    
    # Volume expansion ratio
    volume_avg = df['volume'].rolling(window=M, min_periods=1).mean()
    volume_expansion = df['volume'] / volume_avg
    
    # Volume-expanded returns
    vol_expanded_returns = price_returns * volume_expansion
    
    # Assess Momentum Persistence
    daily_returns = df['close'].pct_change()
    
    # Calculate consecutive positive/negative streaks
    pos_streak = (daily_returns > 0).astype(int)
    neg_streak = (daily_returns < 0).astype(int)
    
    # Compute streak lengths with different weights
    for i in range(1, len(pos_streak)):
        if pos_streak.iloc[i] == 1:
            pos_streak.iloc[i] = pos_streak.iloc[i-1] + 1
        if neg_streak.iloc[i] == 1:
            neg_streak.iloc[i] = neg_streak.iloc[i-1] + 1
    
    # Persistence score with higher weights for longer streaks
    persistence_score = np.where(daily_returns > 0, 
                                pos_streak ** 1.2,  # Positive streaks get higher weight
                                -neg_streak ** 1.1)  # Negative streaks
    
    # Combine Volume Expansion with Persistence
    factor = vol_expanded_returns * persistence_score
    
    # Fill NaN values
    factor = factor.fillna(0)
    
    return factor
