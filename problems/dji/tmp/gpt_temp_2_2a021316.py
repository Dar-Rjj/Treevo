import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10):
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Calculate Intraday Volume Intensity
    previous_close_volume = df['volume'].shift(1)
    intraday_volume_intensity = (df['volume'] - previous_close_volume) / (df['high'] - df['low'])
    
    # Calculate Volume Impact Score
    volume_sum_n_days = df['volume'].rolling(window=n).sum()
    avg_high_low_n_days = (df['high'] - df['low']).rolling(window=n).mean()
    volume_impact_score = volume_sum_n_days / avg_high_low_n_days
    
    # Multiply Intraday Return by Volume Impact Score
    weighted_intraday_return = intraday_return * volume_impact_score
    
    # Calculate Weighted Intraday Volume Intensity
    weighted_intraday_volume_intensity = intraday_volume_intensity * intraday_return
    
    # Calculate Intraday and Breakout Momentum
    intraday_breakout_momentum = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Adjust by Volume
    cumulative_volume = df['volume'].rolling(window=n).sum()
    adjusted_momentum = intraday_breakout_momentun / cumulative_volume
    
    # Combine all the components
    factor = (weighted_intraday_return + weighted_intraday_volume_intensity + adjusted_momentum) / 3
    
    return factor
