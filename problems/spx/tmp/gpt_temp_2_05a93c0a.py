import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate 5-day and 20-day Simple Moving Averages (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    # Calculate Price Volume Trend (PVT)
    df['PVT'] = 0
    for i in range(1, len(df)):
        df.loc[df.index[i], 'PVT'] = df.loc[df.index[i-1], 'PVT'] + (df.loc[df.index[i], 'close'] - df.loc[df.index[i-1], 'close']) * df.loc[df.index[i], 'volume']
    
    # Determine the SMA trend
    df['SMA_trend'] = (df['SMA_5'] > df['SMA_20']).astype(int)
    
    # Determine the PVT momentum
    df['PVT_momentum'] = (df['PVT'].diff() > 0).astype(int)
    
    # Combine SMA trend and PVT momentum to form the alpha factor
    conditions = [
        (df['SMA_trend'] == 1) & (df['PVT_momentum'] == 1),
        (df['SMA_trend'] == 1) & (df['PVT_momentum'] == 0),
        (df['SMA_trend'] == 0) & (df['PVT_momentum'] == 1),
        (df['SMA_trend'] == 0) & (df['PVT_momentum'] == 0)
    ]
    choices = [1, 0.5, -0.5, -1]
    df['alpha_factor'] = pd.np.select(conditions, choices, default=0)
    
    return df['alpha_factor']
