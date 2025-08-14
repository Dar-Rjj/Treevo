import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df, lookback_period=20, momentum_periods=10, smooth_span=5):
    # Calculate Close-to-Low Spread
    close_to_low_spread = (df['close'] - df['low']).clip(lower=0)
    
    # Sum Volume over Period
    accumulated_volume = df['volume'].rolling(window=lookback_period).sum()
    
    # Cumulative Spread Over Period
    accumulated_spread = close_to_low_spread.rolling(window=lookback_period).sum()
    
    # Divide by Accumulated Volume
    volume_adjusted_spread = accumulated_spread / accumulated_volume
    volume_adjusted_spread[accumulated_volume == 0] = 0
    
    # Calculate Momentum
    momentum = df['close'] - df['close'].shift(momentum_periods)
    
    # Calculate Relative Strength
    positive_momentum = momentum[momentum > 0].fillna(0).rolling(window=lookback_period).sum()
    negative_momentum = momentum[momentum < 0].fillna(0).rolling(window=lookback_period).sum()
    relative_strength = positive_momentum / negative_momentum
    relative_strength.replace([pd.np.inf, -pd.np.inf], 0, inplace=True)
    
    # Calculate Final Alpha Factor
    adjusted_relative_strength = relative_strength * volume_adjusted_spread
    final_alpha_factor = adjusted_relative_strength.ewm(span=smooth_span).mean()
    
    return final_alpha_factor
