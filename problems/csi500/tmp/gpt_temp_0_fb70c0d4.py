import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Intraday Return Momentum
    # Compute normalized intraday returns
    daily_range = data['high'] - data['low']
    daily_range = daily_range.replace(0, np.nan)  # Avoid division by zero
    intraday_return = (data['close'] - data['open']) / daily_range
    
    # Calculate intraday momentum persistence (streak of same direction)
    streak = pd.Series(0, index=data.index, dtype=int)
    momentum_strength = pd.Series(0.0, index=data.index)
    
    for i in range(1, len(data)):
        if intraday_return.iloc[i] * intraday_return.iloc[i-1] > 0:
            streak.iloc[i] = streak.iloc[i-1] + 1
        else:
            streak.iloc[i] = 1
        
        # Calculate momentum strength (streak length × average magnitude)
        if streak.iloc[i] >= 1:
            lookback = min(streak.iloc[i], 5)  # Limit lookback to 5 days
            avg_magnitude = intraday_return.iloc[i-lookback+1:i+1].abs().mean()
            momentum_strength.iloc[i] = streak.iloc[i] * avg_magnitude
    
    # 2. Analyze Volume Confirmation Patterns
    # Calculate volume-to-range ratio
    volume_to_range = data['volume'] / daily_range
    volume_to_range = volume_to_range.replace([np.inf, -np.inf], np.nan)
    
    # Compute 5-day average volume-to-range
    vol_range_avg_5d = volume_to_range.rolling(window=5, min_periods=1).mean()
    
    # Volume confirmation signal
    volume_confirmation = volume_to_range / vol_range_avg_5d
    
    # Detect volume breakouts (2× 10-day average)
    vol_range_avg_10d = volume_to_range.rolling(window=10, min_periods=1).mean()
    volume_breakout = volume_to_range > (2 * vol_range_avg_10d)
    
    # 3. Combine Momentum and Volume Signals
    composite_score = momentum_strength * volume_confirmation
    
    # Apply conditional enhancement on volume breakout days
    composite_score = np.where(volume_breakout, composite_score * 2, composite_score)
    
    # 4. Incorporate Price Level Context
    # Calculate relative price position within 20-day range
    low_20d = data['low'].rolling(window=20, min_periods=1).min()
    high_20d = data['high'].rolling(window=20, min_periods=1).max()
    price_range_20d = high_20d - low_20d
    price_range_20d = price_range_20d.replace(0, np.nan)
    
    relative_position = (data['close'] - low_20d) / price_range_20d
    
    # Adjust signal by price position
    adjusted_score = composite_score * (1 + relative_position)
    
    # 5. Apply Directional Consistency Filter
    # Require minimum 3-day consistent intraday direction
    direction_consistent = streak >= 3
    final_factor = np.where(direction_consistent, adjusted_score, 0)
    
    return pd.Series(final_factor, index=data.index)
