import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10):
    # Calculate daily price change using close price
    df['price_change'] = df['close'].diff()
    
    # Initialize rolling sum variable for volume
    df['rolling_volume_sum'] = 0
    
    # Update rolling sum with each day's volume
    for i in range(n, len(df)):
        df.loc[df.index[i], 'rolling_volume_sum'] = df.loc[df.index[i-n:i], 'volume'].sum()
    
    # Assign direction based on price change
    df['direction'] = 0
    df.loc[df['price_change'] > 0, 'direction'] = 1
    df.loc[df['price_change'] < 0, 'direction'] = -1
    
    # Multiply direction by cumulative volume
    df['weighted_volume'] = df['direction'] * df['rolling_volume_sum']
    
    # Sum up all daily direction-weighted volumes
    df['cumulative_volume_momentum'] = df['weighted_volume'].cumsum()
    
    return df['cumulative_volume_momentum']
