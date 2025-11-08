import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Alpha Factor: Intraday Momentum Persistence with Volume Confirmation
    
    Combines persistent intraday momentum signals with volume confirmation
    and adjusts for recent volatility context.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Calculate intraday momentum components
    intraday_return_strength = (df['close'] - df['open']) / (df['high'] - df['low'] + epsilon)
    momentum_direction = np.sign(df['close'] - df['open'])
    momentum_magnitude = np.abs(intraday_return_strength)
    
    # Initialize persistence tracking
    consecutive_days = pd.Series(1, index=df.index)
    persistence_weighted_momentum = pd.Series(0.0, index=df.index)
    
    # Track persistence and calculate weighted momentum
    for i in range(1, len(df)):
        current_date = df.index[i]
        prev_date = df.index[i-1]
        
        # Count consecutive same-direction days
        if momentum_direction[current_date] == momentum_direction[prev_date]:
            consecutive_days[current_date] = consecutive_days[prev_date] + 1
        else:
            consecutive_days[current_date] = 1
        
        # Calculate persistence-weighted momentum with exponential decay
        decay_factor = 0.95 ** (consecutive_days[current_date] - 1)
        persistence_weighted_momentum[current_date] = (consecutive_days[current_date] * 
                                                     momentum_magnitude[current_date] * 
                                                     decay_factor)
    
    # Volume confirmation mechanism
    volume_trend = df['volume'] / df['volume'].shift(1)
    volume_trend.iloc[0] = 1.0  # Handle first day
    
    # Calculate volume confirmation score
    volume_confirmation_score = pd.Series(0, index=df.index, dtype=int)
    
    for i in range(len(df)):
        current_date = df.index[i]
        
        if i == 0:
            volume_confirmation_score[current_date] = 0
            continue
            
        # Positive alignment conditions
        if ((volume_trend[current_date] > 1 and momentum_direction[current_date] > 0) or
            (volume_trend[current_date] < 1 and momentum_direction[current_date] < 0)):
            volume_confirmation_score[current_date] = 1
        # Negative alignment
        else:
            volume_confirmation_score[current_date] = -1
    
    # Combine persistence with volume confirmation
    combined_signal = persistence_weighted_momentum * volume_confirmation_score
    
    # Adjust for recent volatility context (5-day average of price range)
    recent_volatility = (df['high'] - df['low']).rolling(window=5, min_periods=1).mean()
    
    # Final alpha factor with volatility adjustment
    final_alpha = combined_signal / (recent_volatility + epsilon)
    
    return final_alpha
