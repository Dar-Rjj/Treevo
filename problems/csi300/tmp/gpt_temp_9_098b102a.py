import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum
    close = df['close']
    
    # Short-term momentum (3-day)
    mom_short = close.pct_change(periods=3)
    
    # Long-term momentum (10-day)
    mom_long = close.pct_change(periods=10)
    
    # Apply Exponential Decay
    decay_factor = 0.9
    weights = np.array([decay_factor ** i for i in range(5)])[::-1]
    
    # Calculate decayed momentum for short-term
    decayed_mom_short = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            window = mom_short.iloc[i-4:i+1]
            decayed_mom_short.iloc[i] = np.sum(window.values * weights)
        else:
            decayed_mom_short.iloc[i] = mom_short.iloc[i]
    
    # Calculate decayed momentum for long-term
    decayed_mom_long = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            window = mom_long.iloc[i-4:i+1]
            decayed_mom_long.iloc[i] = np.sum(window.values * weights)
        else:
            decayed_mom_long.iloc[i] = mom_long.iloc[i]
    
    # Combined decayed momentum
    decayed_momentum = 0.6 * decayed_mom_short + 0.4 * decayed_mom_long
    
    # Volume Confirmation
    volume = df['volume']
    volume_avg_5d = volume.rolling(window=5, min_periods=1).mean()
    volume_ratio = volume / volume_avg_5d
    
    # Volume-weighted momentum
    volume_weighted_momentum = decayed_momentum * volume_ratio
    
    # Directional Consistency Filter
    directional_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            # Get momentum signs for past 5 days
            momentum_window = decayed_momentum.iloc[i-4:i+1]
            signs = np.sign(momentum_window)
            
            # Count consecutive days with same sign as current
            current_sign = signs.iloc[-1]
            count = 0
            for j in range(len(signs)-1, -1, -1):
                if signs.iloc[j] == current_sign:
                    count += 1
                else:
                    break
            directional_consistency.iloc[i] = count
        else:
            directional_consistency.iloc[i] = 1
    
    # Final factor: Volume-weighted momentum multiplied by directional consistency
    factor = volume_weighted_momentum * directional_consistency
    
    return factor
